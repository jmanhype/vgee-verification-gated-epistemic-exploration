"""
model.py — VGEE Model Wrapper
Verification-Gated Epistemic Exploration (§4)

Wraps a HuggingFace causal LM with:
  - Entropy tracking on every forward pass
  - Trajectory-level uncertainty aggregation
  - Reference model management for KL regularization

Paper §5.1: "We fine-tune a decoder-only transformer (e.g., Llama-3-8B) starting
from a supervised fine-tuning (SFT) checkpoint."

NOTE: This module does NOT re-implement a transformer. It imports and wraps
HuggingFace transformers models.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    GenerationConfig,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .utils import (
    compute_token_entropy,
    compute_trajectory_uncertainty,
    get_reasoning_mask,
    uncertainty_gate,
)


# ---------------------------------------------------------------------------
# §4  VGEEConfig
# ---------------------------------------------------------------------------

@dataclass
class VGEEConfig:
    """
    Configuration for the VGEE wrapper.

    All values map to entries in configs/base.yaml.
    UNSPECIFIED values noted inline.
    """

    # §5.1 Base model identifier [PAPER example]
    base_model_name: str = "meta-llama/Meta-Llama-3-8B"

    # §5.1 Path to SFT checkpoint (or HuggingFace model hub id)
    sft_checkpoint: Optional[str] = None

    # §4.2 Uncertainty threshold δ [UNSPECIFIED — paper: "tuned on validation"]
    # Alternatives: 0.5, 1.0, 1.5, 2.0
    delta: float = 1.0

    # §4.3 Base KL coefficient β_base [UNSPECIFIED]
    # Alternatives: 0.01, 0.04, 0.1
    beta_base: float = 0.04

    # §4.3 Strict KL multiplier κ (Case B) [UNSPECIFIED — paper: "κ >> 1"]
    # Alternatives: 5, 10, 20
    kappa: float = 10.0

    # §5.1 K trajectories per prompt [UNSPECIFIED]
    # Alternatives: 4, 8, 16
    k_trajectories: int = 8

    # Max new tokens for generation [UNSPECIFIED]
    max_new_tokens: int = 1024

    # Mixed precision dtype [UNSPECIFIED]
    # Alternatives: "fp32", "bf16", "fp16"
    torch_dtype: str = "bfloat16"

    # Reference model update strategy [UNSPECIFIED]
    # "frozen": fixed at SFT weights; "ema": exponential moving average
    reference_model_strategy: str = "frozen"

    # EMA decay if reference_model_strategy == "ema" [UNSPECIFIED]
    ema_decay: float = 0.99

    # Uncertainty aggregation method [PAPER Eq 2: "max"]
    uncertainty_aggregation: str = "max"

    # Answer delimiter token string [UNSPECIFIED — paper does not specify]
    # Used to identify reasoning vs answer boundary for U_τ computation
    answer_delimiter: str = "The answer is"


# ---------------------------------------------------------------------------
# §4  VGEEWrapper
# ---------------------------------------------------------------------------

class VGEEWrapper(nn.Module):
    """
    VGEE model wrapper — §4.

    Wraps a HuggingFace causal LM and adds:
      1. Entropy tracking on forward passes (Eq 1)
      2. Trajectory uncertainty aggregation (Eq 2)
      3. Uncertainty gate for verification routing (§4.2)
      4. Reference model management for KL computation (§4.3)

    Paper §5.1: "We use Llama-3-8B as our base model, initialized from a
    supervised fine-tuning (SFT) checkpoint."

    Importantly: this wrapper does not modify the underlying transformer
    architecture. All VGEE logic is applied externally at the trajectory level.
    """

    def __init__(
        self,
        config: VGEEConfig,
        policy_model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """
        Args:
            config:        VGEEConfig with all hyperparameters
            policy_model:  pre-loaded HuggingFace model (if None, loads from config)
            tokenizer:     pre-loaded tokenizer (if None, loads from config)
        """
        super().__init__()
        self.config = config
        self._dtype = _str_to_dtype(config.torch_dtype)

        # Load policy model
        if policy_model is not None:
            self.policy_model = policy_model
        else:
            checkpoint = config.sft_checkpoint or config.base_model_name
            self.policy_model = AutoModelForCausalLM.from_pretrained(
                checkpoint,
                torch_dtype=self._dtype,
                # [UNSPECIFIED] attn_implementation — alternatives: "eager", "sdpa", "flash_attention_2"
                # flash_attention_2 recommended for training efficiency but requires install
            )

        # Load tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            base = config.sft_checkpoint or config.base_model_name
            self.tokenizer = AutoTokenizer.from_pretrained(base)
            if self.tokenizer.pad_token is None:
                # [UNSPECIFIED] pad token choice for decoder-only models
                self.tokenizer.pad_token = self.tokenizer.eos_token

        # Reference model (frozen copy of SFT) for KL computation (§4.3)
        # Paper implies reference model is the SFT checkpoint; see §4.3
        self.reference_model: Optional[PreTrainedModel] = None
        self._init_reference_model()

        # Cache answer delimiter token id for reasoning mask computation
        self._answer_delimiter_ids = self.tokenizer.encode(
            config.answer_delimiter,
            add_special_tokens=False,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_reference_model(self):
        """
        §4.3 — Initialize the reference model.

        Paper implies KL is computed against a fixed reference (SFT checkpoint).
        Strategy is configurable (frozen / EMA) per config.reference_model_strategy.
        """
        # Deep copy policy model weights as the reference starting point
        self.reference_model = copy.deepcopy(self.policy_model)

        # Reference model is never trained — freeze all parameters
        for param in self.reference_model.parameters():
            param.requires_grad_(False)

        self.reference_model.eval()

    def update_reference_model(self):
        """
        Update reference model according to reference_model_strategy.

        §4.3 — Paper does not specify update frequency or strategy in detail.
        [UNSPECIFIED] Called by trainer at configurable intervals.
        """
        if self.config.reference_model_strategy == "frozen":
            # Reference model never changes — stays as SFT checkpoint
            pass

        elif self.config.reference_model_strategy == "ema":
            # [UNSPECIFIED] EMA update: ref ← decay * ref + (1 - decay) * policy
            decay = self.config.ema_decay
            with torch.no_grad():
                for ref_param, policy_param in zip(
                    self.reference_model.parameters(),
                    self.policy_model.parameters(),
                ):
                    ref_param.data.mul_(decay).add_(policy_param.data, alpha=1.0 - decay)

        else:
            raise ValueError(
                f"Unknown reference_model_strategy '{self.config.reference_model_strategy}'. "
                "Use 'frozen' or 'ema'."
            )

    # ------------------------------------------------------------------
    # Forward pass with entropy tracking
    # ------------------------------------------------------------------

    def forward_with_entropy(
        self,
        input_ids: torch.Tensor,       # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
        labels: Optional[torch.Tensor] = None,  # (B, T) for loss computation
    ) -> Dict:
        """
        Forward pass through policy model with per-token entropy tracking.

        Eq 1: H_t = -Σ_v P_θ(v | x_{<t}, y_{<t}) log P_θ(v | x_{<t}, y_{<t})
        Eq 2: U_τ = max_{t ∈ reasoning} H_t

        Args:
            input_ids:      (B, T) — token ids
            attention_mask: (B, T) — 1=valid, 0=pad
            labels:         (B, T) — token ids shifted for LM loss

        Returns:
            dict with keys:
              logits:           (B, T, V) — raw logits
              log_probs:        (B, T, V) — log-probabilities
              token_entropy:    (B, T)    — H_t per Eq 1
              trajectory_uncertainty: (B,) — U_τ per Eq 2
              high_uncertainty_mask:  (B,) bool — U_τ >= δ
              loss:             scalar   — standard LM loss (if labels provided)
              past_key_values:  KV cache
        """
        # Policy model forward pass
        outputs: CausalLMOutputWithPast = self.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=False,
            return_dict=True,
        )

        logits = outputs.logits  # (B, T, V)

        # Compute log-probabilities
        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)

        # Eq 1 — Token-level entropy H_t
        token_entropy = compute_token_entropy(
            logits=logits,
            attention_mask=attention_mask,
        )  # (B, T)

        # Build reasoning mask to restrict U_τ to chain-of-thought tokens (§4.2)
        # [UNSPECIFIED] boundary identification approach
        reasoning_mask = self._build_reasoning_mask(input_ids)  # (B, T)

        # Eq 2 — Trajectory uncertainty U_τ
        U_tau = compute_trajectory_uncertainty(
            token_entropies=token_entropy,
            reasoning_mask=reasoning_mask,
            attention_mask=attention_mask,
            aggregation=self.config.uncertainty_aggregation,
        )  # (B,)

        # §4.2 — Uncertainty gate
        high_uncertainty_mask = uncertainty_gate(U_tau, self.config.delta)  # (B,) bool

        return {
            "logits": logits,                              # (B, T, V)
            "log_probs": log_probs,                        # (B, T, V)
            "token_entropy": token_entropy,                # (B, T)
            "trajectory_uncertainty": U_tau,               # (B,)
            "high_uncertainty_mask": high_uncertainty_mask,  # (B,) bool
            "loss": outputs.loss,                          # scalar or None
            "past_key_values": outputs.past_key_values,
        }

    def forward_reference(
        self,
        input_ids: torch.Tensor,       # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """
        Forward pass through frozen reference model.

        Used for KL divergence computation in §4.3.

        Returns:
            ref_log_probs: (B, T, V) — log-probs from reference model
        """
        assert self.reference_model is not None, "Reference model not initialized"

        with torch.no_grad():
            ref_outputs = self.reference_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        ref_log_probs = F.log_softmax(ref_outputs.logits, dim=-1)  # (B, T, V)
        return ref_log_probs  # (B, T, V)

    # ------------------------------------------------------------------
    # Trajectory generation with entropy tracking
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_trajectories(
        self,
        prompt_input_ids: torch.Tensor,       # (1, prompt_len) or (B, prompt_len)
        prompt_attention_mask: torch.Tensor,  # (1, prompt_len) or (B, prompt_len)
        k: Optional[int] = None,              # number of trajectories K per prompt
        generation_config: Optional[GenerationConfig] = None,
    ) -> Dict:
        """
        §4.2 — Sample K trajectories per prompt.

        Paper §5.1: "We sample K trajectories per prompt during rollout."
        K is UNSPECIFIED; defaults to config.k_trajectories.

        For each trajectory we compute:
          - Generated token ids
          - Per-token log-probabilities under policy
          - Per-token entropy H_t (Eq 1)
          - Trajectory uncertainty U_τ (Eq 2)

        Args:
            prompt_input_ids:       (B, prompt_len) or (1, prompt_len)
            prompt_attention_mask:  (B, prompt_len) or (1, prompt_len)
            k:                      number of trajectories (overrides config.k_trajectories)
            generation_config:      HuggingFace GenerationConfig

        Returns:
            dict with keys:
              trajectory_ids:        (B*K, full_len) — full token sequences
              prompt_len:            int
              policy_log_probs:      (B*K, gen_len, V) — per-token log-probs
              token_entropies:       (B*K, gen_len)    — H_t values (Eq 1)
              trajectory_uncertainty:(B*K,)             — U_τ values (Eq 2)
              high_uncertainty_mask: (B*K,) bool        — U_τ >= δ
              attention_masks:       (B*K, full_len)
        """
        K = k or self.config.k_trajectories
        B, prompt_len = prompt_input_ids.shape

        # Expand prompts K times along batch dimension for parallel sampling
        # (B, prompt_len) → (B*K, prompt_len)
        expanded_ids = prompt_input_ids.repeat_interleave(K, dim=0)         # (B*K, P)
        expanded_mask = prompt_attention_mask.repeat_interleave(K, dim=0)   # (B*K, P)

        # Build generation config
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                # [UNSPECIFIED] temperature for sampling during rollout
                # Alternatives: 0.7, 0.8, 1.0
                temperature=1.0,
                # [UNSPECIFIED] top-p nucleus sampling
                # Alternatives: 0.9, 0.95, 1.0
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Generate full sequences (prompt + trajectory)
        generated = self.policy_model.generate(
            input_ids=expanded_ids,
            attention_mask=expanded_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,  # needed to recover per-step logits
        )

        # generated.sequences:  (B*K, prompt_len + gen_len)
        # generated.scores:     tuple of (B*K, V) tensors, length = gen_len
        full_ids = generated.sequences  # (B*K, full_len)
        full_len = full_ids.shape[1]

        # Reconstruct attention mask for full sequences
        full_mask = (full_ids != self.tokenizer.pad_token_id).long()  # (B*K, full_len)

        # Stack per-step logits into a single tensor
        # generated.scores is a tuple of gen_len tensors each (B*K, V)
        gen_len = full_len - prompt_len
        if generated.scores:
            stacked_logits = torch.stack(generated.scores, dim=1)  # (B*K, gen_len, V)
        else:
            # Fallback: re-run forward pass to get logits
            # [UNSPECIFIED] This path triggers if output_scores not working
            stacked_logits = self._recompute_logits(full_ids, full_mask, prompt_len)

        # Per-token log-probabilities for generated portion only
        policy_log_probs = F.log_softmax(stacked_logits, dim=-1)  # (B*K, gen_len, V)

        # Per-token entropy H_t over generated portion (Eq 1)
        gen_attention_mask = full_mask[:, prompt_len:]  # (B*K, gen_len)
        token_entropies = compute_token_entropy(
            logits=stacked_logits,
            attention_mask=gen_attention_mask,
        )  # (B*K, gen_len)

        # Eq 2 — Trajectory uncertainty U_τ over generated portion
        U_tau = compute_trajectory_uncertainty(
            token_entropies=token_entropies,
            attention_mask=gen_attention_mask,
            aggregation=self.config.uncertainty_aggregation,
        )  # (B*K,)

        # §4.2 — Uncertainty gate
        high_uncertainty_mask = uncertainty_gate(U_tau, self.config.delta)  # (B*K,) bool

        return {
            "trajectory_ids": full_ids,                          # (B*K, full_len)
            "prompt_len": prompt_len,
            "policy_log_probs": policy_log_probs,                # (B*K, gen_len, V)
            "token_entropies": token_entropies,                  # (B*K, gen_len)
            "trajectory_uncertainty": U_tau,                     # (B*K,)
            "high_uncertainty_mask": high_uncertainty_mask,      # (B*K,) bool
            "attention_masks": full_mask,                        # (B*K, full_len)
            "gen_len": gen_len,
        }

    def _recompute_logits(
        self,
        input_ids: torch.Tensor,   # (B, T)
        attention_mask: torch.Tensor,  # (B, T)
        prompt_len: int,
    ) -> torch.Tensor:
        """
        Fallback: re-run full forward pass to obtain per-token logits for generated portion.

        Returns:
            logits: (B, gen_len, V)
        """
        with torch.no_grad():
            outputs = self.policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        # Slice off prompt portion; shift by 1 (predict next token)
        # logits[:, prompt_len-1:-1, :] predicts tokens at positions prompt_len..T-1
        return outputs.logits[:, prompt_len - 1: -1, :]  # (B, gen_len, V)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_reasoning_mask(
        self,
        input_ids: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        """
        §4.2 — Build mask for reasoning tokens (exclude final answer portion).

        [UNSPECIFIED] Paper does not describe segmentation strategy.
        We use the configured answer_delimiter to find the boundary.

        Returns:
            reasoning_mask: (B, T) bool
        """
        B, T = input_ids.shape
        mask = torch.ones(B, T, dtype=torch.bool, device=input_ids.device)

        # Find first occurrence of answer_delimiter token sequence
        delim_ids = self._answer_delimiter_ids
        if not delim_ids:
            return mask  # no delimiter — treat all as reasoning

        for b in range(B):
            seq = input_ids[b].tolist()
            # Search for delimiter subsequence
            for t in range(len(seq) - len(delim_ids) + 1):
                if seq[t: t + len(delim_ids)] == delim_ids:
                    mask[b, t:] = False
                    break

        return mask  # (B, T) bool

    def get_policy_model(self) -> PreTrainedModel:
        """Return the underlying policy model for optimizer registration."""
        return self.policy_model

    def train(self, mode: bool = True):
        """Set policy model to train mode; reference model always stays eval."""
        self.policy_model.train(mode)
        if self.reference_model is not None:
            self.reference_model.eval()
        return self

    def eval(self):
        self.policy_model.eval()
        return self

    def parameters(self, recurse: bool = True):
        """Only policy model parameters are trainable."""
        return self.policy_model.parameters(recurse=recurse)

    def save_pretrained(self, path: str):
        """Save policy model and tokenizer."""
        self.policy_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[VGEEConfig] = None,
    ) -> "VGEEWrapper":
        """Load a saved VGEE policy checkpoint."""
        if config is None:
            config = VGEEConfig(sft_checkpoint=checkpoint_path)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            torch_dtype=_str_to_dtype(config.torch_dtype),
        )
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        return cls(config=config, policy_model=model, tokenizer=tokenizer)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype name to torch.dtype."""
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Use one of: {list(mapping)}")
    return mapping[dtype_str]


def load_vgee_model_from_config(cfg: dict) -> VGEEWrapper:
    """
    Convenience factory: build VGEEWrapper from a parsed YAML config dict.

    Args:
        cfg: parsed configs/base.yaml as dict

    Returns:
        VGEEWrapper instance
    """
    vgee_config = VGEEConfig(
        base_model_name=cfg["model"]["base_model_name"],
        sft_checkpoint=cfg["model"].get("sft_checkpoint"),
        delta=cfg["training"]["delta"],
        beta_base=cfg["kl_regularizer"]["beta_base"],
        kappa=cfg["kl_regularizer"]["kappa"],
        k_trajectories=cfg["training"]["k_trajectories"],
        max_new_tokens=cfg["training"]["max_new_tokens"],
        torch_dtype=cfg["model"].get("mixed_precision", "bfloat16"),
        reference_model_strategy=cfg["kl_regularizer"]["reference_model_strategy"],
        ema_decay=cfg["kl_regularizer"]["ema_decay"],
    )
    return VGEEWrapper(config=vgee_config)
