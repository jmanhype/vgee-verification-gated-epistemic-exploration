"""
train.py — VGEE Training Loop  [PRIMARY DELIVERABLE]
Verification-Gated Epistemic Exploration: Resolving the Winner-Take-All Paradox in RLVR

This file IS the core contribution of the paper (type-b: new training method).

Paper §4–§5: VGEE training procedure:
  1. For each batch of prompts:
     a. Sample K trajectories per prompt  (§4.2)
     b. Compute token entropy H_t (Eq 1) and trajectory uncertainty U_τ (Eq 2)
     c. Route high-uncertainty trajectories to external verifier (§4.2)
     d. Assign cases A/B/C based on uncertainty + verification result (§4.3)
     e. Compute PPO advantage with GAE: γ=0.99, λ=0.95 (§5.1)
     f. Update policy with Eq 5: L(θ) = L^CLIP - c_1 L^VF + Conditional_KL(θ)

  2. Repeat for 10 iterations (§5.1)

Training specs (§5.1):
  - Model: decoder-only transformer (Llama-3-8B) from SFT checkpoint
  - Optimizer: Adam, lr=1e-5, cosine decay
  - Batch size: 512 prompts
  - K trajectories per prompt [UNSPECIFIED]
  - γ=0.99, λ=0.95
  - 10 training iterations
"""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# VGEE modules
from .data import (
    ReasoningDataset,
    Trajectory,
    assign_rewards,
    build_token_reward_tensor,
    collect_trajectories_from_output,
    create_dataloader,
    TASK_VERIFIER,
)
from .loss import (
    VGEELossOutput,
    compute_gae,
    vgee_loss,
)
from .model import VGEEConfig, VGEEWrapper
from .utils import batch_verify

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# §5.1  Value Head
# ---------------------------------------------------------------------------

class ValueHead(nn.Module):
    """
    Scalar value head attached to the policy LM.

    §5.1 / Eq 5: L^VF requires V_θ(s_t) predictions.

    [UNSPECIFIED] Paper does not describe the value head architecture.
    Standard approach: linear projection from LM hidden states.

    Architecture:
        hidden_states (B, T, D) → Linear(D, 1) → squeeze → (B, T)
    """

    def __init__(
        self,
        hidden_size: int,   # D — hidden dimension of base LM
        # [UNSPECIFIED] value head initialization
        # Alternatives: zero-init, small random, kaiming
        init_std: float = 0.02,  # [UNSPECIFIED]
    ):
        super().__init__()
        # [UNSPECIFIED] Whether to use LayerNorm before projection
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1, bias=False)

        # Small initialization to avoid disrupting early training
        nn.init.normal_(self.linear.weight, std=init_std)

    def forward(
        self,
        hidden_states: torch.Tensor,  # (B, T, D) — from LM backbone
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (B, T, D)

        Returns:
            values: (B, T) — scalar value estimates
        """
        normed = self.norm(hidden_states)       # (B, T, D)
        values = self.linear(normed).squeeze(-1)  # (B, T)
        return values  # (B, T)


class VGEEModelWithValueHead(nn.Module):
    """
    Combines VGEEWrapper (policy) with a ValueHead for PPO training.

    §5.1 — PPO requires both policy and value function.
    Value head is trained jointly with policy [UNSPECIFIED — paper implies this].
    """

    def __init__(
        self,
        vgee_wrapper: VGEEWrapper,
        hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.vgee = vgee_wrapper

        # Infer hidden size from policy model config if not provided
        if hidden_size is None:
            hidden_size = vgee_wrapper.policy_model.config.hidden_size

        self.value_head = ValueHead(hidden_size=hidden_size)

    def forward_policy_and_value(
        self,
        input_ids: torch.Tensor,        # (B, T)
        attention_mask: torch.Tensor,   # (B, T)
        labels: Optional[torch.Tensor] = None,
    ) -> Dict:
        """
        Forward pass returning both policy outputs and value estimates.

        Returns:
            dict with all keys from VGEEWrapper.forward_with_entropy plus:
              values: (B, T) — value head predictions
        """
        # We need hidden states for value head — override policy model call
        outputs = self.vgee.policy_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = outputs.logits              # (B, T, V)
        hidden_states = outputs.hidden_states[-1]  # (B, T, D) — last layer

        # Value predictions
        values = self.value_head(hidden_states)  # (B, T)

        # Entropy tracking (mirrors VGEEWrapper.forward_with_entropy)
        import torch.nn.functional as F
        from .utils import compute_token_entropy, compute_trajectory_uncertainty, uncertainty_gate

        log_probs = F.log_softmax(logits, dim=-1)  # (B, T, V)
        token_entropy = compute_token_entropy(
            logits=logits, attention_mask=attention_mask
        )  # (B, T)

        reasoning_mask = self.vgee._build_reasoning_mask(input_ids)  # (B, T)
        U_tau = compute_trajectory_uncertainty(
            token_entropies=token_entropy,
            reasoning_mask=reasoning_mask,
            attention_mask=attention_mask,
            aggregation=self.vgee.config.uncertainty_aggregation,
        )  # (B,)
        high_uncertainty_mask = uncertainty_gate(U_tau, self.vgee.config.delta)  # (B,)

        return {
            "logits": logits,                                  # (B, T, V)
            "log_probs": log_probs,                            # (B, T, V)
            "token_entropy": token_entropy,                    # (B, T)
            "trajectory_uncertainty": U_tau,                   # (B,)
            "high_uncertainty_mask": high_uncertainty_mask,    # (B,) bool
            "values": values,                                  # (B, T)
            "loss": outputs.loss,
        }

    def parameters(self, recurse: bool = True):
        """Training parameters: policy model + value head."""
        return list(self.vgee.policy_model.parameters(recurse)) + \
               list(self.value_head.parameters(recurse))

    def train(self, mode: bool = True):
        self.vgee.train(mode)
        self.value_head.train(mode)
        return self

    def eval(self):
        self.vgee.eval()
        self.value_head.eval()
        return self


# ---------------------------------------------------------------------------
# §5.1  Rollout Buffer
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """
    Stores collected trajectories and derived tensors for PPO updates.

    §5.1: "We collect rollouts then perform PPO update steps."

    [UNSPECIFIED] Buffer size, truncation behavior.
    """

    def __init__(self):
        self.trajectories: List[Trajectory] = []
        self.advantages: Optional[torch.Tensor] = None   # (N, T)
        self.returns: Optional[torch.Tensor] = None      # (N, T)
        self.old_log_probs: Optional[torch.Tensor] = None  # (N, T)
        self.old_values: Optional[torch.Tensor] = None     # (N, T)

    def clear(self):
        self.trajectories = []
        self.advantages = None
        self.returns = None
        self.old_log_probs = None
        self.old_values = None

    def __len__(self):
        return len(self.trajectories)


# ---------------------------------------------------------------------------
# §4  / §5.1  Verification Orchestration
# ---------------------------------------------------------------------------

def run_verification_step(
    trajectories: List[Trajectory],
    majority_vote_k: int = 5,          # [PAPER §5.1]: BBH uses 5 majority vote samples
    interpreter_timeout: int = 10,     # [UNSPECIFIED]
    bbh_extra_samples: Optional[List[List[str]]] = None,  # pre-sampled BBH extras
) -> List[Trajectory]:
    """
    §4.2 — Route high-uncertainty trajectories to external verifier.

    Paper §4.2: "High-uncertainty trajectories (U_τ ≥ δ) are sent to external
    verification. Low-uncertainty trajectories are not verified."

    Paper §5.1:
      - MATH/GSM8K: Python interpreter
      - BBH: majority voting with 5 samples

    Args:
        trajectories:        list of all trajectories in batch
        majority_vote_k:     number of BBH extra samples [PAPER §5.1: 5]
        interpreter_timeout: seconds for Python eval [UNSPECIFIED]
        bbh_extra_samples:   pre-generated extra samples for BBH majority vote

    Returns:
        trajectories with verified_correct field set:
          - High-uncertainty → True/False based on verifier
          - Low-uncertainty  → None (not verified; Case C)
    """
    # Separate by uncertainty gate result
    high_unc_indices = [i for i, t in enumerate(trajectories) if t.is_high_uncertainty]
    low_unc_indices = [i for i, t in enumerate(trajectories) if not t.is_high_uncertainty]

    logger.debug(
        f"Verification gate: {len(high_unc_indices)} high-uncertainty, "
        f"{len(low_unc_indices)} low-uncertainty trajectories"
    )

    # Low-uncertainty trajectories → Case C (no verification)
    for idx in low_unc_indices:
        trajectories[idx].verified_correct = None

    if not high_unc_indices:
        return trajectories

    # Group by task type for batch verification
    by_task: Dict[str, List[int]] = defaultdict(list)
    for idx in high_unc_indices:
        by_task[trajectories[idx].task_type].append(idx)

    for task_type, indices in by_task.items():
        problems = [trajectories[i].prompt for i in indices]
        gen_answers = [trajectories[i].trajectory_text for i in indices]
        ground_truths = [trajectories[i].ground_truth for i in indices]

        if task_type in ("math", "gsm8k"):
            # §5.1: Python interpreter verification
            results = batch_verify(
                problems=problems,
                generated_answers=gen_answers,
                ground_truths=ground_truths,
                task_type=task_type,
                timeout_seconds=interpreter_timeout,
            )
        elif task_type == "bbh":
            # §5.1: Majority voting with 5 samples
            if bbh_extra_samples is not None:
                mv_samples = [bbh_extra_samples[i] for i in indices]
            else:
                # Fallback: use the single generated answer as the only sample
                # [UNSPECIFIED] What to do when extra BBH samples aren't available
                logger.warning(
                    "BBH verification called without pre-sampled extra trajectories. "
                    "Falling back to single-sample verification (not ideal)."
                )
                mv_samples = [[gen_answers[j]] for j in range(len(indices))]

            results = batch_verify(
                problems=problems,
                generated_answers=gen_answers,
                ground_truths=ground_truths,
                task_type=task_type,
                majority_vote_samples=mv_samples,
            )
        else:
            # [UNSPECIFIED] Unknown task types default to False
            logger.warning(f"Unknown task type '{task_type}' for verification. Defaulting to False.")
            results = [False] * len(indices)

        for idx, result in zip(indices, results):
            trajectories[idx].verified_correct = result

    return trajectories


# ---------------------------------------------------------------------------
# §5.1  PPO Mini-batch Update
# ---------------------------------------------------------------------------

def ppo_minibatch_update(
    model_with_vh: VGEEModelWithValueHead,
    batch_input_ids: torch.Tensor,        # (B, T) — full sequences (prompt+gen)
    batch_attention_mask: torch.Tensor,   # (B, T)
    batch_old_log_probs: torch.Tensor,    # (B, T) — from rollout (policy log-prob per token)
    batch_old_values: torch.Tensor,       # (B, T) — value estimates from rollout
    batch_advantages: torch.Tensor,       # (B, T) — GAE advantages
    batch_returns: torch.Tensor,          # (B, T) — GAE returns
    batch_high_uncertainty: torch.Tensor, # (B,) bool
    batch_verified_correct: torch.Tensor, # (B,) bool
    optimizer: torch.optim.Optimizer,
    cfg: dict,                            # parsed base.yaml
    prompt_len: int,                      # to slice off prompt portion
    scaler: Optional[torch.cuda.amp.GradScaler] = None,  # for mixed precision
) -> Dict:
    """
    §5.1 — Single PPO mini-batch gradient update.

    Eq 5: L(θ) = E_t [L^CLIP(θ) - c_1 L^VF(θ) + Conditional_KL(θ)]

    Args:
        model_with_vh:          VGEEModelWithValueHead
        batch_input_ids:        (B, T) — full token sequences
        batch_attention_mask:   (B, T)
        batch_old_log_probs:    (B, T) — old policy log-probs per token
        batch_old_values:       (B, T) — old value estimates
        batch_advantages:       (B, T) — GAE advantages
        batch_returns:          (B, T) — GAE returns
        batch_high_uncertainty: (B,) bool
        batch_verified_correct: (B,) bool
        optimizer:              torch optimizer
        cfg:                    config dict
        prompt_len:             int — length of prompt portion (excluded from loss)
        scaler:                 optional AMP GradScaler

    Returns:
        info dict with all loss components
    """
    use_amp = scaler is not None
    dtype = torch.bfloat16 if cfg["model"].get("mixed_precision") == "bf16" else torch.float32

    with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
        # Forward pass through policy + value head
        fwd = model_with_vh.forward_policy_and_value(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
        )

        # Policy log-probs for generated tokens only
        # Slice generated portion: positions [prompt_len-1 : T-1] predict [prompt_len : T]
        # (standard language model convention)
        gen_log_probs_full = fwd["log_probs"][:, prompt_len - 1: -1, :]  # (B, gen_len, V)
        gen_attention_mask = batch_attention_mask[:, prompt_len:]          # (B, gen_len)

        # Per-token log-probs of actually selected tokens
        # Need actual token ids for the generated portion
        gen_token_ids = batch_input_ids[:, prompt_len:]  # (B, gen_len)
        gen_log_probs_token = gen_log_probs_full.gather(
            dim=-1,
            index=gen_token_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, gen_len)

        # Reference model log-probs
        ref_log_probs_full = model_with_vh.vgee.forward_reference(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
        )[:, prompt_len - 1: -1, :]  # (B, gen_len, V)

        # Value predictions for generated portion
        values_pred = fwd["values"][:, prompt_len - 1: -1]  # (B, gen_len)

        # Slice old tensors to generated portion
        old_lp_gen = batch_old_log_probs[:, :gen_token_ids.shape[1]]    # (B, gen_len)
        old_val_gen = batch_old_values[:, :gen_token_ids.shape[1]]       # (B, gen_len)
        adv_gen = batch_advantages[:, :gen_token_ids.shape[1]]           # (B, gen_len)
        ret_gen = batch_returns[:, :gen_token_ids.shape[1]]              # (B, gen_len)

        # Eq 5 — VGEE full loss
        loss_output: VGEELossOutput = vgee_loss(
            policy_log_probs_token=gen_log_probs_token,
            policy_log_probs_full=gen_log_probs_full,
            ref_log_probs_full=ref_log_probs_full,
            old_log_probs_token=old_lp_gen,
            values_pred=values_pred,
            values_old=old_val_gen,
            advantages=adv_gen,
            returns=ret_gen,
            attention_mask=gen_attention_mask,
            high_uncertainty_mask=batch_high_uncertainty,
            verified_correct=batch_verified_correct,
            beta_base=cfg["kl_regularizer"]["beta_base"],
            kappa=cfg["kl_regularizer"]["kappa"],
            c_1=cfg["training"]["value_function_coeff"],
            clip_epsilon=cfg["training"]["ppo_clip_epsilon"],
        )

    # Gradient update
    optimizer.zero_grad()

    if scaler is not None:
        scaler.scale(loss_output.total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model_with_vh.parameters(),
            cfg["training"]["max_grad_norm"],
        )
        scaler.step(optimizer)
        scaler.update()
    else:
        loss_output.total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model_with_vh.parameters(),
            cfg["training"]["max_grad_norm"],
        )
        optimizer.step()

    return loss_output.info


# ---------------------------------------------------------------------------
# §5.1  Full Rollout Collection
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_rollout(
    model_with_vh: VGEEModelWithValueHead,
    batch: Dict,
    cfg: dict,
    device: torch.device,
    bbh_extra_samples: Optional[List[List[str]]] = None,
) -> Tuple[RolloutBuffer, Dict]:
    """
    §4.2 / §5.1 — Collect rollout trajectories for one prompt batch.

    Steps:
      1. Sample K trajectories per prompt
      2. Compute H_t (Eq 1), U_τ (Eq 2)
      3. Run verification gate (§4.2)
      4. Assign rewards
      5. Compute GAE advantages

    Args:
        model_with_vh:      model + value head
        batch:              collated DataLoader batch
        cfg:                config dict
        device:             torch device
        bbh_extra_samples:  extra samples for BBH majority voting

    Returns:
        buffer:  populated RolloutBuffer
        info:    diagnostic dict
    """
    model_with_vh.eval()

    input_ids = batch["input_ids"].to(device)          # (B, P)
    attention_mask = batch["attention_mask"].to(device) # (B, P)
    problems = batch["problems"]
    ground_truths = batch["answers"]
    task_types = batch["task_types"]

    B = input_ids.shape[0]
    K = cfg["training"]["k_trajectories"]

    # Step 1: Sample K trajectories per prompt (§4.2, §5.1)
    gen_output = model_with_vh.vgee.generate_trajectories(
        prompt_input_ids=input_ids,
        prompt_attention_mask=attention_mask,
        k=K,
    )

    # Step 2: Build Trajectory objects with H_t and U_τ
    trajectories = collect_trajectories_from_output(
        model_output=gen_output,
        tokenizer=model_with_vh.vgee.tokenizer,
        problems=problems,
        ground_truths=ground_truths,
        task_types=task_types,
        k=K,
    )  # list of B*K Trajectory objects

    # Step 3: Verification gate (§4.2)
    # Route high-uncertainty trajectories to external verifier
    trajectories = run_verification_step(
        trajectories=trajectories,
        majority_vote_k=cfg["verification"]["majority_vote_samples"],
        interpreter_timeout=cfg["verification"]["interpreter_timeout"],
        bbh_extra_samples=bbh_extra_samples,
    )

    # Step 4: Assign rewards based on verification results
    # [UNSPECIFIED] reward values; using {-1, +1} as default
    verification_results = [
        t.verified_correct if t.verified_correct is not None
        else False  # Case C: low-uncertainty, not verified — treat as no reward
        for t in trajectories
    ]
    trajectories = assign_rewards(
        trajectories=trajectories,
        verification_results=verification_results,
    )

    # Step 5: Get value estimates for rollout sequences
    # Re-run forward pass with value head
    traj_ids = gen_output["trajectory_ids"]           # (B*K, full_len)
    traj_masks = gen_output["attention_masks"]        # (B*K, full_len)
    prompt_len = gen_output["prompt_len"]

    # Forward through value head to get V(s_t)
    vh_outputs = model_with_vh.forward_policy_and_value(
        input_ids=traj_ids.to(device),
        attention_mask=traj_masks.to(device),
    )
    old_values_full = vh_outputs["values"]  # (B*K, full_len)
    old_log_probs_full = vh_outputs["log_probs"]  # (B*K, full_len, V)

    # Per-token log-probs for generated tokens
    gen_ids = traj_ids[:, prompt_len:]  # (B*K, gen_len)
    old_log_probs_gen = old_log_probs_full[:, prompt_len - 1: -1, :]  # (B*K, gen_len, V)
    old_lp_token = old_log_probs_gen.gather(
        dim=-1,
        index=gen_ids.unsqueeze(-1).to(device)
    ).squeeze(-1)  # (B*K, gen_len)

    old_values_gen = old_values_full[:, prompt_len - 1: -1]  # (B*K, gen_len)

    # Step 5: Compute per-token rewards tensor
    reward_tensor = build_token_reward_tensor(trajectories, device=device)  # (B*K, gen_len)

    # Step 6: GAE — compute advantages and returns
    # §5.1: γ=0.99, λ=0.95 [PAPER]
    gen_attn_mask = traj_masks[:, prompt_len:].to(device)  # (B*K, gen_len)
    advantages, returns = compute_gae(
        rewards=reward_tensor,
        values=old_values_gen,
        attention_mask=gen_attn_mask,
        gamma=cfg["training"]["gae_gamma"],
        lam=cfg["training"]["gae_lambda"],
    )  # both (B*K, gen_len)

    # Populate buffer
    buffer = RolloutBuffer()
    buffer.trajectories = trajectories
    buffer.advantages = advantages.cpu()          # (B*K, gen_len)
    buffer.returns = returns.cpu()                # (B*K, gen_len)
    buffer.old_log_probs = old_lp_token.cpu()    # (B*K, gen_len)
    buffer.old_values = old_values_gen.cpu()     # (B*K, gen_len)

    # Store full sequences and masks back on trajectories for PPO update
    for i, traj in enumerate(trajectories):
        traj.values = old_values_gen[i].cpu()

    rollout_info = {
        "n_trajectories": len(trajectories),
        "n_high_uncertainty": sum(t.is_high_uncertainty for t in trajectories),
        "n_verified_correct": sum(
            1 for t in trajectories
            if t.verified_correct is True
        ),
        "mean_uncertainty": sum(t.trajectory_uncertainty for t in trajectories) / len(trajectories),
        "mean_reward": sum(t.reward for t in trajectories if t.reward is not None) / max(
            sum(1 for t in trajectories if t.reward is not None), 1
        ),
    }

    return buffer, rollout_info, gen_output


# ---------------------------------------------------------------------------
# §5.1  Main VGEE Training Loop  — PRIMARY CONTRIBUTION
# ---------------------------------------------------------------------------

def train_vgee(
    cfg: dict,
    model_with_vh: VGEEModelWithValueHead,
    train_dataset: ReasoningDataset,
    eval_datasets: Optional[Dict[str, ReasoningDataset]] = None,
    device: Optional[torch.device] = None,
    resume_from: Optional[str] = None,
) -> Dict:
    """
    VGEE Training Loop — §4 and §5.1  [PRIMARY DELIVERABLE]

    This function implements the full VGEE training procedure as described
    in the paper. It is the core algorithmic contribution.

    Algorithm (paraphrased from paper §4):
      for iteration in range(num_iterations):  # §5.1: 10 iterations
          for batch in dataloader:             # §5.1: 512 prompts/batch
              # Rollout phase
              for k in range(K):               # sample K trajectories
                  generate trajectory τ_k
                  compute H_t (Eq 1)
                  compute U_τ (Eq 2)
                  if U_τ >= δ:               # §4.2 uncertainty gate
                      verify(τ_k)            # external verifier

              # Assign cases A/B/C (§4.3)
              # Compute rewards and GAE advantages

              # PPO update phase
              for epoch in range(ppo_epochs):  # [UNSPECIFIED]
                  for mini_batch in rollout:
                      compute L(θ) per Eq 5
                      update θ

    Paper §5.1 specs:
      - lr=1e-5 with cosine decay
      - γ=0.99, λ=0.95
      - batch_size=512
      - 10 iterations

    Args:
        cfg:               parsed configs/base.yaml
        model_with_vh:     VGEEModelWithValueHead
        train_dataset:     training ReasoningDataset
        eval_datasets:     optional dict of task_name → eval ReasoningDataset
        device:            torch device (auto-detected if None)
        resume_from:       checkpoint path to resume from

    Returns:
        training_metrics: dict of final metrics
    """
    # Setup
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    model_with_vh = model_with_vh.to(device)
    model_with_vh.vgee.reference_model = model_with_vh.vgee.reference_model.to(device)

    # §5.1 — Optimizer: Adam with cosine LR decay [PAPER]
    optimizer = AdamW(
        model_with_vh.parameters(),
        lr=cfg["optimizer"]["lr"],                # 1e-5 [PAPER]
        betas=(
            cfg["optimizer"]["adam_beta1"],        # [UNSPECIFIED]
            cfg["optimizer"]["adam_beta2"],        # [UNSPECIFIED]
        ),
        eps=cfg["optimizer"]["adam_epsilon"],      # [UNSPECIFIED]
        weight_decay=cfg["optimizer"]["weight_decay"],  # [UNSPECIFIED]
    )

    # §5.1 — Cosine LR decay [PAPER]
    num_iterations = cfg["training"]["num_iterations"]  # 10 [PAPER]
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_iterations,
        eta_min=0.0,  # [UNSPECIFIED] minimum LR — decay to 0
    )

    # Mixed precision scaler
    use_amp = cfg["model"].get("mixed_precision") in ("bf16", "fp16")
    # Note: bf16 autocast doesn't use GradScaler (only fp16 does)
    scaler = (
        torch.cuda.amp.GradScaler()
        if use_amp and cfg["model"].get("mixed_precision") == "fp16"
        else None
    )

    # DataLoader — §5.1: batch_size=512 [PAPER]
    train_loader = create_dataloader(
        dataset=train_dataset,
        tokenizer=model_with_vh.vgee.tokenizer,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
    )

    # Output directory
    output_dir = cfg["logging"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Optional: WandB logging [UNSPECIFIED — not described in paper]
    wandb_run = None
    if cfg["logging"].get("wandb_enabled", False):
        try:
            import wandb
            wandb_run = wandb.init(
                project=cfg["logging"]["wandb_project"],
                config=cfg,
            )
        except ImportError:
            logger.warning("wandb not installed — skipping W&B logging.")

    # Resume from checkpoint if provided
    start_iteration = 0
    if resume_from is not None:
        checkpoint = torch.load(os.path.join(resume_from, "checkpoint.pt"))
        model_with_vh.vgee.policy_model.load_state_dict(checkpoint["policy_state_dict"])
        model_with_vh.value_head.load_state_dict(checkpoint["value_head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_iteration = checkpoint["iteration"] + 1
        logger.info(f"Resumed from iteration {start_iteration}")

    all_metrics = []

    # =========================================================================
    # §5.1  VGEE MAIN TRAINING LOOP — 10 iterations [PAPER]
    # =========================================================================
    for iteration in range(start_iteration, num_iterations):
        iter_start = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"VGEE Iteration {iteration + 1}/{num_iterations}")
        logger.info(f"{'='*60}")

        model_with_vh.train()
        iteration_metrics = defaultdict(list)

        # ------------------------------------------------------------------
        # Rollout + PPO update over all batches in the iteration
        # ------------------------------------------------------------------
        for batch_idx, batch in enumerate(train_loader):
            batch_start = time.time()

            # ==============================================================
            # PHASE 1: ROLLOUT COLLECTION (§4.2)
            # Sample K trajectories, compute uncertainty, verify high-U_τ
            # ==============================================================
            logger.info(f"  Batch {batch_idx}: collecting rollout...")

            buffer, rollout_info, gen_output = collect_rollout(
                model_with_vh=model_with_vh,
                batch=batch,
                cfg=cfg,
                device=device,
            )

            logger.info(
                f"  Rollout: {rollout_info['n_trajectories']} trajectories, "
                f"{rollout_info['n_high_uncertainty']} high-uncertainty "
                f"({rollout_info['n_verified_correct']} verified correct), "
                f"mean_reward={rollout_info['mean_reward']:.3f}, "
                f"mean_U_τ={rollout_info['mean_uncertainty']:.4f}"
            )

            for k, v in rollout_info.items():
                iteration_metrics[f"rollout/{k}"].append(v)

            # ==============================================================
            # PHASE 2: PPO UPDATE (§5.1, Eq 5)
            # Multiple epochs over the collected rollout mini-batches
            # ==============================================================
            model_with_vh.train()

            # Prepare tensors from buffer
            traj_ids = gen_output["trajectory_ids"]    # (B*K, full_len)
            traj_masks = gen_output["attention_masks"] # (B*K, full_len)
            prompt_len = gen_output["prompt_len"]

            high_unc = torch.tensor(
                [t.is_high_uncertainty for t in buffer.trajectories],
                dtype=torch.bool,
            )  # (B*K,)

            verified = torch.tensor(
                [
                    t.verified_correct if t.verified_correct is not None else False
                    for t in buffer.trajectories
                ],
                dtype=torch.bool,
            )  # (B*K,)

            N = len(buffer.trajectories)
            mini_batch_size = cfg["training"]["mini_batch_size"]  # [UNSPECIFIED]
            ppo_epochs = cfg["training"]["ppo_epochs"]            # [UNSPECIFIED]

            for ppo_epoch in range(ppo_epochs):
                # Shuffle indices for mini-batch sampling
                perm = torch.randperm(N)

                for mb_start in range(0, N, mini_batch_size):
                    mb_end = min(mb_start + mini_batch_size, N)
                    mb_idx = perm[mb_start:mb_end]

                    # Gather mini-batch tensors
                    mb_input_ids = traj_ids[mb_idx].to(device)          # (mb, full_len)
                    mb_attn_mask = traj_masks[mb_idx].to(device)        # (mb, full_len)
                    mb_old_lp = buffer.old_log_probs[mb_idx].to(device) # (mb, gen_len)
                    mb_old_val = buffer.old_values[mb_idx].to(device)   # (mb, gen_len)
                    mb_adv = buffer.advantages[mb_idx].to(device)       # (mb, gen_len)
                    mb_ret = buffer.returns[mb_idx].to(device)          # (mb, gen_len)
                    mb_high_unc = high_unc[mb_idx].to(device)           # (mb,)
                    mb_verified = verified[mb_idx].to(device)           # (mb,)

                    # Eq 5 — VGEE gradient update
                    update_info = ppo_minibatch_update(
                        model_with_vh=model_with_vh,
                        batch_input_ids=mb_input_ids,
                        batch_attention_mask=mb_attn_mask,
                        batch_old_log_probs=mb_old_lp,
                        batch_old_values=mb_old_val,
                        batch_advantages=mb_adv,
                        batch_returns=mb_ret,
                        batch_high_uncertainty=mb_high_unc,
                        batch_verified_correct=mb_verified,
                        optimizer=optimizer,
                        cfg=cfg,
                        prompt_len=prompt_len,
                        scaler=scaler,
                    )

                    for k, v in update_info.items():
                        iteration_metrics[f"ppo/{k}"].append(v)

                    logger.debug(
                        f"    PPO epoch {ppo_epoch+1}/{ppo_epochs}, "
                        f"mb {mb_start//mini_batch_size}: "
                        f"loss={update_info['total_loss']:.4f}, "
                        f"clip={update_info['ppo_clip_loss']:.4f}, "
                        f"kl={update_info['conditional_kl_loss']:.4f}"
                    )

            batch_elapsed = time.time() - batch_start
            logger.info(
                f"  Batch {batch_idx} done in {batch_elapsed:.1f}s | "
                f"loss={_mean(iteration_metrics['ppo/total_loss']):.4f} | "
                f"Case A: {int(_sum(iteration_metrics['rollout/n_high_uncertainty']))} | "
                f"KL: {_mean(iteration_metrics['ppo/conditional_kl_loss']):.4f}"
            )

            # Clear rollout buffer
            buffer.clear()

        # Step LR scheduler [PAPER §5.1: cosine decay]
        scheduler.step()

        # Update reference model if using EMA [UNSPECIFIED — configurable]
        model_with_vh.vgee.update_reference_model()

        # ------------------------------------------------------------------
        # Evaluation (§5.2)
        # ------------------------------------------------------------------
        eval_metrics = {}
        if (
            eval_datasets is not None
            and (iteration + 1) % cfg["evaluation"]["eval_every_n_iterations"] == 0
        ):
            logger.info(f"  Running evaluation...")
            from .evaluate import evaluate_vgee
            model_with_vh.eval()
            eval_metrics = evaluate_vgee(
                model_with_vh=model_with_vh,
                eval_datasets=eval_datasets,
                cfg=cfg,
                device=device,
            )
            for task, metrics in eval_metrics.items():
                logger.info(f"  {task}: {metrics}")

        # ------------------------------------------------------------------
        # Checkpointing
        # ------------------------------------------------------------------
        if (iteration + 1) % cfg["logging"]["save_every_n_iterations"] == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint_iter_{iteration+1}")
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(
                {
                    "iteration": iteration,
                    "policy_state_dict": model_with_vh.vgee.policy_model.state_dict(),
                    "value_head_state_dict": model_with_vh.value_head.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "config": cfg,
                },
                os.path.join(ckpt_dir, "checkpoint.pt"),
            )
            model_with_vh.vgee.save_pretrained(ckpt_dir)
            logger.info(f"  Checkpoint saved to {ckpt_dir}")

        # ------------------------------------------------------------------
        # Iteration summary
        # ------------------------------------------------------------------
        iter_elapsed = time.time() - iter_start
        iter_summary = {
            "iteration": iteration + 1,
            "elapsed_seconds": iter_elapsed,
            "lr": scheduler.get_last_lr()[0],
            **{k: _mean(v) for k, v in iteration_metrics.items()},
            **{f"eval/{t}/{m}": v for t, metrics in eval_metrics.items() for m, v in metrics.items()},
        }
        all_metrics.append(iter_summary)

        logger.info(
            f"Iteration {iteration+1} summary: "
            f"elapsed={iter_elapsed:.0f}s, "
            f"lr={iter_summary['lr']:.2e}, "
            f"mean_loss={iter_summary.get('ppo/total_loss', 0):.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(iter_summary, step=iteration + 1)

    # =========================================================================
    # Training complete
    # =========================================================================
    final_dir = os.path.join(output_dir, "final_model")
    model_with_vh.vgee.save_pretrained(final_dir)
    logger.info(f"\nVGEE training complete. Final model saved to {final_dir}")

    if wandb_run is not None:
        wandb_run.finish()

    return {"all_metrics": all_metrics}


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    """
    CLI entry point for VGEE training.

    Usage:
        python -m src.train --config configs/base.yaml --data_path /path/to/data.json
    """
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description="VGEE Training")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--data_path", required=True, help="Path to training data JSON")
    parser.add_argument("--task_type", default="math", choices=["math", "gsm8k", "bbh"])
    parser.add_argument("--device", default=None, help="cuda or cpu")
    parser.add_argument("--resume_from", default=None, help="Checkpoint path to resume")
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device) if args.device else None

    # Build model
    from .model import load_vgee_model_from_config
    vgee_wrapper = load_vgee_model_from_config(cfg)
    model_with_vh = VGEEModelWithValueHead(vgee_wrapper)

    # Build dataset
    from .data import ReasoningDataset
    train_ds = ReasoningDataset(
        task_type=args.task_type,
        split="train",
        data_path=args.data_path,
        max_samples=args.max_samples,
    )

    # Train
    metrics = train_vgee(
        cfg=cfg,
        model_with_vh=model_with_vh,
        train_dataset=train_ds,
        device=device,
        resume_from=args.resume_from,
    )

    print("Training complete.")
    print(f"Final metrics: {metrics['all_metrics'][-1]}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mean(lst):
    return sum(lst) / len(lst) if lst else 0.0

def _sum(lst):
    return sum(lst) if lst else 0.0


if __name__ == "__main__":
    main()
