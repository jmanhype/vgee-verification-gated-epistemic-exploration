"""
data.py — VGEE Dataset and Trajectory Collection
Verification-Gated Epistemic Exploration (§5.1)

Implements:
  - Dataset skeleton for MATH, GSM8K, BBH  [PAPER §5.2]
  - Trajectory collection with entropy tracking
  - Trajectory dataclass used throughout training

Paper §5.2: "We evaluate on MATH500, GSM8K, and BBH."
Paper §5.1: "We sample K trajectories per prompt for PPO rollout."
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer


# ---------------------------------------------------------------------------
# §5.2  Task Types
# ---------------------------------------------------------------------------

TASK_MATH = "math"
TASK_GSM8K = "gsm8k"
TASK_BBH = "bbh"

SUPPORTED_TASKS = {TASK_MATH, TASK_GSM8K, TASK_BBH}

# Verification strategy per task [PAPER §5.1]
TASK_VERIFIER = {
    TASK_MATH: "python_interpreter",
    TASK_GSM8K: "python_interpreter",
    TASK_BBH: "majority_voting",
}


# ---------------------------------------------------------------------------
# §4.2 / §5.1  Trajectory Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """
    A single sampled trajectory τ from the policy.

    Used to store all data needed for:
      - Uncertainty computation (Eq 2)
      - Verification gate routing (§4.2)
      - PPO update (Eq 5)

    §5.1: "K trajectories are sampled per prompt."
    """
    # Problem and ground truth
    prompt: str                    # original problem string
    ground_truth: str              # correct answer
    task_type: str                 # "math", "gsm8k", or "bbh"

    # Generated trajectory
    trajectory_text: str           # full generated text (chain-of-thought + answer)
    trajectory_ids: torch.Tensor   # (full_len,)   token ids

    # Log-probs and entropy [Eq 1]
    policy_log_probs: torch.Tensor  # (gen_len, V) log π_θ distribution
    token_entropies: torch.Tensor   # (gen_len,)   H_t per Eq 1

    # Uncertainty [Eq 2]
    trajectory_uncertainty: float   # U_τ scalar

    # Uncertainty gate [§4.2]
    is_high_uncertainty: bool       # U_τ >= δ

    # Verification result [§4.2] — filled in after verification step
    verified_correct: Optional[bool] = None

    # Value function estimates — filled in during rollout
    values: Optional[torch.Tensor] = None  # (gen_len,)

    # Reward — filled in after verification
    reward: Optional[float] = None

    # Metadata
    prompt_len: int = 0
    trajectory_idx: int = 0    # index within the K trajectories for this prompt
    prompt_idx: int = 0        # index of the source prompt in the batch


# ---------------------------------------------------------------------------
# §5.2  Dataset Base Class
# ---------------------------------------------------------------------------

class ReasoningDataset(Dataset):
    """
    Base class for reasoning task datasets used in VGEE.

    Paper §5.2: "We train and evaluate on MATH500, GSM8K, and BBH."

    Concrete subclasses implement:
      - _load_data(): load from disk or HuggingFace datasets
      - format_prompt(): convert (problem, task_type) → prompt string
    """

    def __init__(
        self,
        task_type: str,
        split: str = "train",           # "train" | "test" | "validation"
        data_path: Optional[str] = None, # local JSON file path
        max_samples: Optional[int] = None,  # truncate dataset [UNSPECIFIED]
        prompt_template: str = "default",   # prompt formatting style [UNSPECIFIED]
    ):
        assert task_type in SUPPORTED_TASKS, \
            f"task_type must be one of {SUPPORTED_TASKS}, got '{task_type}'"

        self.task_type = task_type
        self.split = split
        self.data_path = data_path
        self.max_samples = max_samples
        self.prompt_template = prompt_template

        # Load raw data
        self.data: List[Dict[str, Any]] = self._load_data()

        if max_samples is not None:
            self.data = self.data[:max_samples]

    def _load_data(self) -> List[Dict[str, Any]]:
        """
        Load dataset from local file or HuggingFace datasets.

        [UNSPECIFIED] Paper does not specify data source format.
        Expected JSON format: [{"problem": str, "answer": str}, ...]

        Subclasses should override this for task-specific loading.
        """
        if self.data_path is not None and os.path.exists(self.data_path):
            with open(self.data_path, "r") as f:
                return json.load(f)

        # Fallback: attempt HuggingFace datasets load
        # [UNSPECIFIED] exact dataset version used in paper
        try:
            from datasets import load_dataset  # type: ignore
            return self._load_from_hf()
        except ImportError:
            raise RuntimeError(
                "datasets library not installed. Install with: pip install datasets\n"
                "Or provide data_path pointing to a JSON file."
            )

    def _load_from_hf(self) -> List[Dict[str, Any]]:
        """
        Load from HuggingFace datasets hub.

        [UNSPECIFIED] Paper does not specify exact HuggingFace dataset IDs.
        Using standard community versions.
        """
        from datasets import load_dataset  # type: ignore

        if self.task_type == TASK_MATH:
            # [UNSPECIFIED] MATH500 split ID
            ds = load_dataset("hendrycks/competition_math", split=self.split)
            return [{"problem": ex["problem"], "answer": ex["solution"]} for ex in ds]

        elif self.task_type == TASK_GSM8K:
            ds = load_dataset("gsm8k", "main", split=self.split)
            return [{"problem": ex["question"], "answer": ex["answer"]} for ex in ds]

        elif self.task_type == TASK_BBH:
            # BBH has multiple sub-tasks; load all or specific
            # [UNSPECIFIED] which BBH sub-tasks the paper uses
            ds = load_dataset("lukaemon/bbh", "boolean_expressions", split="test")
            return [{"problem": ex["input"], "answer": ex["target"]} for ex in ds]

        else:
            raise ValueError(f"Unknown task_type: {self.task_type}")

    def format_prompt(self, problem: str) -> str:
        """
        Format a problem into a prompt string.

        §5.1: Paper does not specify the exact prompt template used.
        [UNSPECIFIED] We use a simple task-specific template.
        Alternatives: few-shot, chain-of-thought instructed, task-specific system prompt.
        """
        if self.task_type == TASK_MATH:
            return (
                f"Solve the following math problem step by step.\n\n"
                f"Problem: {problem}\n\n"
                f"Solution: Let me think through this carefully.\n"
            )
        elif self.task_type == TASK_GSM8K:
            return (
                f"Solve the following grade school math problem step by step.\n\n"
                f"Question: {problem}\n\n"
                f"Answer: Let me work through this.\n"
            )
        elif self.task_type == TASK_BBH:
            return (
                f"Answer the following reasoning question.\n\n"
                f"Question: {problem}\n\n"
                f"Answer: Let me reason step by step.\n"
            )
        else:
            return f"Question: {problem}\nAnswer:"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        item = self.data[idx]
        return {
            "problem": item["problem"],
            "answer": item.get("answer", item.get("solution", "")),
            "prompt": self.format_prompt(item["problem"]),
            "task_type": self.task_type,
        }


# ---------------------------------------------------------------------------
# Specialized Dataset Subclasses
# ---------------------------------------------------------------------------

class MATHDataset(ReasoningDataset):
    """MATH competition dataset — §5.2."""
    def __init__(self, split: str = "train", **kwargs):
        super().__init__(task_type=TASK_MATH, split=split, **kwargs)


class GSM8KDataset(ReasoningDataset):
    """GSM8K grade school math dataset — §5.2."""
    def __init__(self, split: str = "train", **kwargs):
        super().__init__(task_type=TASK_GSM8K, split=split, **kwargs)


class BBHDataset(ReasoningDataset):
    """BIG-Bench Hard dataset — §5.2."""
    def __init__(self, split: str = "test", **kwargs):
        # [PAPER §5.2] BBH is typically used as evaluation only (no train split)
        super().__init__(task_type=TASK_BBH, split=split, **kwargs)


# ---------------------------------------------------------------------------
# §5.1  Collate Function and DataLoader
# ---------------------------------------------------------------------------

def collate_reasoning_batch(
    batch: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizer,
    max_prompt_len: int = 512,  # [UNSPECIFIED] max prompt length
    padding_side: str = "left",  # [UNSPECIFIED] left-pad for decoder-only
) -> Dict[str, Any]:
    """
    Collate a list of reasoning examples into a padded batch.

    §5.1: "Batch size: 512 prompts."

    Args:
        batch:          list of dicts from ReasoningDataset.__getitem__
        tokenizer:      HuggingFace tokenizer
        max_prompt_len: maximum prompt token length [UNSPECIFIED]
        padding_side:   "left" for decoder-only models [UNSPECIFIED]

    Returns:
        dict with:
          input_ids:      (B, max_len) — padded prompt token ids
          attention_mask: (B, max_len)
          problems:       list of str
          answers:        list of str
          task_types:     list of str
          prompts:        list of str
    """
    prompts = [ex["prompt"] for ex in batch]
    problems = [ex["problem"] for ex in batch]
    answers = [ex["answer"] for ex in batch]
    task_types = [ex["task_type"] for ex in batch]

    # Tokenize prompts
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side

    encoded = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_len,
        return_tensors="pt",
    )

    tokenizer.padding_side = original_padding_side

    return {
        "input_ids": encoded["input_ids"],          # (B, P)
        "attention_mask": encoded["attention_mask"], # (B, P)
        "problems": problems,
        "answers": answers,
        "task_types": task_types,
        "prompts": prompts,
    }


def create_dataloader(
    dataset: ReasoningDataset,
    tokenizer: PreTrainedTokenizer,
    batch_size: int = 512,  # [PAPER §5.1]
    shuffle: bool = True,
    num_workers: int = 4,   # [UNSPECIFIED]
    max_prompt_len: int = 512,  # [UNSPECIFIED]
) -> DataLoader:
    """
    Create a DataLoader for VGEE training.

    Paper §5.1: "Batch size: 512 prompts."

    Args:
        dataset:        ReasoningDataset instance
        tokenizer:      HuggingFace tokenizer
        batch_size:     number of prompts per batch [PAPER]
        shuffle:        whether to shuffle [UNSPECIFIED]
        num_workers:    DataLoader workers [UNSPECIFIED]
        max_prompt_len: max prompt token length [UNSPECIFIED]

    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=lambda b: collate_reasoning_batch(b, tokenizer, max_prompt_len),
        pin_memory=True,
        drop_last=True,  # [UNSPECIFIED] drop incomplete batches for stable training
    )


# ---------------------------------------------------------------------------
# §5.1  Trajectory Collection
# ---------------------------------------------------------------------------

def collect_trajectories_from_output(
    model_output: Dict,        # output from VGEEWrapper.generate_trajectories
    tokenizer: PreTrainedTokenizer,
    problems: List[str],       # (B,) problem strings
    ground_truths: List[str],  # (B,) ground truth answers
    task_types: List[str],     # (B,) task type strings
    k: int,                    # K trajectories per prompt
) -> List[Trajectory]:
    """
    Convert raw generation output into a list of Trajectory objects.

    §5.1: "We collect K trajectories per prompt and compute entropy/uncertainty
    for each trajectory."

    Args:
        model_output:  dict from VGEEWrapper.generate_trajectories
        tokenizer:     for decoding token ids back to text
        problems:      original problem strings
        ground_truths: correct answers
        task_types:    per-problem task type
        k:             number of trajectories per prompt

    Returns:
        trajectories: list of Trajectory objects (length B*K)
    """
    trajectory_ids = model_output["trajectory_ids"]        # (B*K, full_len)
    prompt_len = model_output["prompt_len"]
    policy_log_probs = model_output["policy_log_probs"]    # (B*K, gen_len, V)
    token_entropies = model_output["token_entropies"]      # (B*K, gen_len)
    U_tau = model_output["trajectory_uncertainty"]         # (B*K,)
    high_uncertainty_mask = model_output["high_uncertainty_mask"]  # (B*K,) bool

    B = len(problems)
    BK = trajectory_ids.shape[0]
    assert BK == B * k, f"Expected {B*k} trajectories, got {BK}"

    trajectories = []
    for bk_idx in range(BK):
        b_idx = bk_idx // k
        traj_idx = bk_idx % k

        # Decode full trajectory (generated portion only)
        gen_ids = trajectory_ids[bk_idx, prompt_len:]  # (gen_len,)
        trajectory_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        traj = Trajectory(
            prompt=problems[b_idx],
            ground_truth=ground_truths[b_idx],
            task_type=task_types[b_idx],
            trajectory_text=trajectory_text,
            trajectory_ids=trajectory_ids[bk_idx],              # (full_len,)
            policy_log_probs=policy_log_probs[bk_idx],          # (gen_len, V)
            token_entropies=token_entropies[bk_idx],            # (gen_len,)
            trajectory_uncertainty=U_tau[bk_idx].item(),
            is_high_uncertainty=high_uncertainty_mask[bk_idx].item(),
            verified_correct=None,   # filled later
            values=None,             # filled by value head
            reward=None,             # filled after verification
            prompt_len=prompt_len,
            trajectory_idx=traj_idx,
            prompt_idx=b_idx,
        )
        trajectories.append(traj)

    return trajectories


def assign_rewards(
    trajectories: List[Trajectory],
    verification_results: List[bool],
    reward_correct: float = 1.0,   # [UNSPECIFIED] reward for correct answer
    reward_wrong: float = -1.0,    # [UNSPECIFIED] reward for wrong answer
                                   # Alternatives: 0.0 for wrong; sparse {0, 1}
) -> List[Trajectory]:
    """
    §4.2 / §5.1 — Assign scalar rewards to trajectories based on verification.

    [UNSPECIFIED] Paper implies binary reward (correct/incorrect) but does not
    state exact reward values.
    Common choices: {0, 1}, {-1, +1}, {0, +1}.

    Args:
        trajectories:        list of Trajectory objects
        verification_results: list of bool (True=correct)
        reward_correct:      reward for verified correct [UNSPECIFIED]
        reward_wrong:        reward for verified incorrect [UNSPECIFIED]

    Returns:
        trajectories with reward and verified_correct fields set
    """
    for traj, is_correct in zip(trajectories, verification_results):
        traj.verified_correct = is_correct
        traj.reward = reward_correct if is_correct else reward_wrong

    return trajectories


def build_token_reward_tensor(
    trajectories: List[Trajectory],
    device: torch.device,
) -> torch.Tensor:
    """
    Convert per-trajectory scalar rewards into per-token reward tensors.

    §5.1 — PPO assigns reward to final token; all other tokens get 0.

    [UNSPECIFIED] Token-level reward distribution strategy.
    Paper uses PPO which typically places reward only at the end of sequence.
    Alternatives: distribute evenly, credit assignment models.

    Args:
        trajectories: list of Trajectory with reward set
        device:       target device

    Returns:
        reward_tensor: (B, T) — per-token rewards, non-zero only at final valid token
    """
    # Find max gen_len for padding
    max_len = max(t.token_entropies.shape[0] for t in trajectories)
    B = len(trajectories)

    reward_tensor = torch.zeros(B, max_len, device=device)  # (B, T)

    for i, traj in enumerate(trajectories):
        T = traj.token_entropies.shape[0]
        if traj.reward is not None:
            # Place reward at the last generated token [UNSPECIFIED reward position]
            reward_tensor[i, T - 1] = traj.reward

    return reward_tensor  # (B, T)
