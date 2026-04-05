"""
utils.py — VGEE Core Utilities
Verification-Gated Epistemic Exploration (§4)

Implements:
  - Token-level entropy H_t  (Eq 1)
  - Trajectory uncertainty U_τ  (Eq 2)
  - Verification gate logic  (§4.2)

All tensor operations annotated with shape comments.
All UNSPECIFIED design choices flagged inline.
"""

from __future__ import annotations

import ast
import contextlib
import io
import math
import multiprocessing
import re
import signal
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# §4.1  Token-Level Entropy
# ---------------------------------------------------------------------------

def compute_token_entropy(
    logits: torch.Tensor,  # (batch, seq_len, vocab_size)  raw model logits
    attention_mask: Optional[torch.Tensor] = None,  # (batch, seq_len) 1=valid 0=pad
    temperature: float = 1.0,  # [UNSPECIFIED] paper does not specify sampling temperature
) -> torch.Tensor:
    """
    Eq 1 — Token-level entropy:
        H_t = -Σ_v P_θ(v | x_{<t}, y_{<t}) log P_θ(v | x_{<t}, y_{<t})

    Paper (§4.1): "We compute token-level entropy over the model's output distribution
    at each reasoning step."

    Args:
        logits:          (B, T, V) — raw logits from the model
        attention_mask:  (B, T)    — 1 for real tokens, 0 for padding
        temperature:     float     — softmax temperature (not specified in paper)

    Returns:
        H:  (B, T)  — per-token entropy in nats
    """
    # Scale logits by temperature before softmax
    # temperature=1.0 gives standard distribution; paper does not specify otherwise
    scaled_logits = logits / temperature  # (B, T, V)

    # Compute log-probabilities via log_softmax for numerical stability
    log_probs = F.log_softmax(scaled_logits, dim=-1)  # (B, T, V)

    # Recover probabilities from log-probs
    probs = log_probs.exp()  # (B, T, V)

    # Eq 1: H_t = -Σ_v p(v) log p(v)
    # Using entropy = -Σ p * log_p  (nats, base e)
    H = -(probs * log_probs).sum(dim=-1)  # (B, T)

    # Zero-out padded positions so they don't affect downstream max/mean
    if attention_mask is not None:
        H = H * attention_mask.float()  # (B, T)

    return H  # (B, T)


def compute_token_entropy_from_probs(
    probs: torch.Tensor,  # (batch, seq_len, vocab_size) already-softmaxed probabilities
    attention_mask: Optional[torch.Tensor] = None,  # (batch, seq_len)
    epsilon: float = 1e-10,  # numerical stability floor
) -> torch.Tensor:
    """
    Variant of Eq 1 when probabilities are already computed (e.g. during rollout).

    Args:
        probs:          (B, T, V) — probability distribution at each step
        attention_mask: (B, T)    — padding mask
        epsilon:        float     — floor to avoid log(0)

    Returns:
        H:  (B, T)  — per-token entropy in nats
    """
    log_probs = (probs + epsilon).log()  # (B, T, V) — safe log
    H = -(probs * log_probs).sum(dim=-1)  # (B, T)

    if attention_mask is not None:
        H = H * attention_mask.float()  # (B, T)

    return H  # (B, T)


# ---------------------------------------------------------------------------
# §4.2  Trajectory Uncertainty
# ---------------------------------------------------------------------------

def compute_trajectory_uncertainty(
    token_entropies: torch.Tensor,  # (batch, seq_len)  per-token H_t values
    reasoning_mask: Optional[torch.Tensor] = None,  # (batch, seq_len) 1=reasoning tokens
    attention_mask: Optional[torch.Tensor] = None,  # (batch, seq_len) 1=valid
    aggregation: str = "max",  # [PAPER Eq 2 uses max] alternatives: "mean", "percentile_95"
) -> torch.Tensor:
    """
    Eq 2 — Trajectory uncertainty:
        U_τ = max_{t ∈ reasoning} H_t

    Paper (§4.2): "We define the trajectory uncertainty U_τ as the maximum token-level
    entropy over the reasoning portion of the trajectory."

    The "reasoning portion" is the chain-of-thought before the final answer.
    Paper does not specify how this segment is identified; we accept a reasoning_mask.

    [UNSPECIFIED] Aggregation: paper explicitly uses max (Eq 2). Alternatives: mean,
    95th-percentile. This implementation supports all via `aggregation` param.

    Args:
        token_entropies: (B, T)   — H_t from compute_token_entropy
        reasoning_mask:  (B, T)   — 1 for reasoning tokens, 0 for answer/pad
                                    If None, uses all valid tokens.
        attention_mask:  (B, T)   — padding mask applied after reasoning_mask
        aggregation:     str      — "max" (paper default), "mean", "percentile_95"

    Returns:
        U_tau:  (B,)  — scalar uncertainty per trajectory
    """
    # Combine masks: restrict to reasoning tokens only
    if reasoning_mask is not None:
        mask = reasoning_mask  # (B, T)
    else:
        # If no reasoning_mask provided, use all valid (non-pad) positions
        # [UNSPECIFIED] — paper does not specify fallback behavior
        mask = torch.ones_like(token_entropies, dtype=torch.bool)  # (B, T)

    if attention_mask is not None:
        mask = mask & attention_mask.bool()  # (B, T)

    # Replace masked positions with -inf for max, or 0 for mean
    if aggregation == "max":
        # Eq 2: U_τ = max_{t ∈ reasoning} H_t
        masked_H = token_entropies.masked_fill(~mask, float('-inf'))  # (B, T)
        U_tau = masked_H.max(dim=-1).values  # (B,)

        # Guard: if a trajectory has no reasoning tokens, uncertainty is 0
        U_tau = torch.where(
            mask.any(dim=-1),
            U_tau,
            torch.zeros_like(U_tau)
        )  # (B,)

    elif aggregation == "mean":
        # [UNSPECIFIED] alternative: mean entropy over reasoning tokens
        masked_H = token_entropies * mask.float()  # (B, T)
        token_count = mask.float().sum(dim=-1).clamp(min=1)  # (B,)
        U_tau = masked_H.sum(dim=-1) / token_count  # (B,)

    elif aggregation == "percentile_95":
        # [UNSPECIFIED] alternative: 95th percentile — robust to single outlier spikes
        U_tau_list = []
        for b in range(token_entropies.shape[0]):
            valid = token_entropies[b][mask[b]]  # (n_valid,)
            if valid.numel() == 0:
                U_tau_list.append(torch.tensor(0.0, device=token_entropies.device))
            else:
                U_tau_list.append(torch.quantile(valid, 0.95))
        U_tau = torch.stack(U_tau_list)  # (B,)

    else:
        raise ValueError(f"Unknown aggregation '{aggregation}'. Use 'max', 'mean', or 'percentile_95'.")

    return U_tau  # (B,)


# ---------------------------------------------------------------------------
# §4.2  Verification Gate
# ---------------------------------------------------------------------------

@dataclass
class VerificationResult:
    """Result from the verification oracle for a single trajectory."""
    trajectory_idx: int           # index within current batch
    is_high_uncertainty: bool     # U_τ >= δ
    verified_correct: Optional[bool]  # None if not sent to verifier
    uncertainty: float            # U_τ scalar value
    verifier_used: Optional[str]  # "python_interpreter" | "majority_voting" | None


def uncertainty_gate(
    U_tau: torch.Tensor,  # (B,)  trajectory uncertainties
    delta: float,         # threshold δ — [UNSPECIFIED] paper says "tuned on validation"
) -> torch.Tensor:
    """
    §4.2  Uncertainty gate: classify trajectories as high/low uncertainty.

        high_uncertainty = U_τ >= δ

    Paper: "We define an uncertainty threshold δ... trajectories where U_τ ≥ δ
    are routed to external verification."

    Args:
        U_tau:  (B,)  — trajectory uncertainties
        delta:  float — threshold δ

    Returns:
        high_uncertainty_mask: (B,) bool — True where U_τ >= δ
    """
    return U_tau >= delta  # (B,) bool


# ---------------------------------------------------------------------------
# Verifier backends  [PAPER §5.1]
# ---------------------------------------------------------------------------

def _timeout_handler(signum, frame):
    raise TimeoutError("Verification timed out")


def verify_math_python_interpreter(
    problem: str,           # natural language problem statement
    generated_answer: str,  # model's generated answer string
    ground_truth: str,      # ground truth answer string
    timeout_seconds: int = 10,  # [UNSPECIFIED] paper does not specify timeout
) -> bool:
    """
    §5.1  Python interpreter verification for MATH / GSM8K.

    Paper: "For MATH and GSM8K, we use a Python interpreter to verify correctness
    by executing the generated solution and comparing to ground truth."

    Strategy:
      1. Try exact string match (stripped/lowercased)
      2. Try numeric comparison (float equality with tolerance)
      3. Try symbolic eval of simple expressions

    [UNSPECIFIED] The paper does not detail the answer extraction/normalization
    pipeline. We implement a best-effort extraction.

    Args:
        problem:          raw problem text (unused in base impl, available for context)
        generated_answer: model's full generated text including chain-of-thought
        ground_truth:     ground truth answer
        timeout_seconds:  max seconds for interpreter execution

    Returns:
        bool — True if verified correct
    """
    pred = _extract_final_answer(generated_answer)
    gt = ground_truth.strip()

    # Step 1: exact string match (case-insensitive, whitespace-normalized)
    if pred.lower().replace(" ", "") == gt.lower().replace(" ", ""):
        return True

    # Step 2: numeric comparison with tolerance
    try:
        pred_num = float(_eval_numeric(pred))
        gt_num = float(_eval_numeric(gt))
        if math.isclose(pred_num, gt_num, rel_tol=1e-4, abs_tol=1e-8):
            return True
    except (ValueError, TypeError, SyntaxError, ZeroDivisionError):
        pass

    # Step 3: symbolic expression execution
    try:
        result = _safe_exec_expression(pred, timeout_seconds)
        gt_result = _safe_exec_expression(gt, timeout_seconds)
        if result is not None and gt_result is not None:
            if math.isclose(float(result), float(gt_result), rel_tol=1e-4):
                return True
    except Exception:
        pass

    return False


def _extract_final_answer(text: str) -> str:
    """
    Extract the final answer from model-generated text.

    [UNSPECIFIED] Paper does not describe answer extraction. Common approaches:
      - "The answer is X" pattern
      - LaTeX \\boxed{X}
      - Last line of output
    """
    # Try LaTeX \\boxed{...}
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Try "the answer is X" or "= X" at end
    answer_match = re.search(
        r'(?:the answer is|answer:|therefore[,:]?|=)\s*([^\n.]+)',
        text,
        re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1).strip()

    # Fallback: last non-empty line
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    return lines[-1] if lines else text.strip()


def _eval_numeric(expr: str) -> Optional[float]:
    """Safely evaluate a simple numeric expression string."""
    # Strip common LaTeX/formatting
    clean = re.sub(r'[,$%]', '', expr).strip()
    clean = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', clean)

    # Only allow safe characters
    if re.fullmatch(r'[\d\s\.\+\-\*/\(\)]+', clean):
        return eval(clean)  # nosec — restricted character set
    return None


def _safe_exec_expression(
    expr: str,
    timeout_seconds: int,
) -> Optional[float]:
    """Execute a string expression in a sandboxed namespace with timeout."""
    # [UNSPECIFIED] Paper describes "Python interpreter" but not sandboxing approach
    allowed_builtins = {"abs": abs, "round": round, "int": int, "float": float}

    # Use signal-based timeout on Unix; on Windows this silently skips
    if hasattr(signal, 'SIGALRM'):
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_seconds)
    try:
        stdout_capture = io.StringIO()
        with contextlib.redirect_stdout(stdout_capture):
            result = eval(expr, {"__builtins__": allowed_builtins}, {})  # nosec
        return result
    except Exception:
        return None
    finally:
        if hasattr(signal, 'SIGALRM'):
            signal.alarm(0)


def verify_bbh_majority_voting(
    problem: str,                  # BBH problem
    model_samples: List[str],      # N samples from the model (paper: N=5)
    ground_truth: str,             # ground truth answer
) -> bool:
    """
    §5.1  Majority voting verification for BBH.

    Paper: "For BBH, we use majority voting with 5 samples as the verification oracle."

    The majority-voted answer is compared against ground truth.
    If majority vote matches ground truth, the trajectory is "verified correct."

    [UNSPECIFIED] It is unclear whether the trajectory being verified is one of the
    5 majority-vote samples, or a separate trajectory. We assume:
      - model_samples are K additional samples drawn for the purpose of verification
      - We return True if majority_vote(model_samples) == ground_truth

    Args:
        problem:       BBH problem text
        model_samples: list of answer strings (paper §5.1: 5 samples)
        ground_truth:  correct answer

    Returns:
        bool — True if majority vote matches ground truth
    """
    if not model_samples:
        return False

    # Extract final answers from each sample
    answers = [_extract_final_answer(s).strip().lower() for s in model_samples]

    # Majority vote: pick most common answer
    from collections import Counter
    vote_counts = Counter(answers)
    majority_answer = vote_counts.most_common(1)[0][0]

    return majority_answer == ground_truth.strip().lower()


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def get_reasoning_mask(
    input_ids: torch.Tensor,  # (B, T)
    answer_start_token_id: int,  # token id that begins the "answer" portion
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Build a boolean mask indicating reasoning (chain-of-thought) tokens.

    §4.2: "We restrict U_τ computation to reasoning tokens, not the final answer."

    [UNSPECIFIED] Paper does not describe how reasoning vs answer tokens are
    demarcated. Common approaches:
      - Special delimiter token (e.g., <answer>, \\boxed)
      - Fixed split point
      - Separate head for each segment

    This implementation uses a sentinel token to mark the answer boundary.

    Args:
        input_ids:             (B, T) — token ids
        answer_start_token_id: int    — id of the delimiter marking answer start
        pad_token_id:          int    — id of the padding token

    Returns:
        reasoning_mask: (B, T) bool — True for reasoning positions
    """
    B, T = input_ids.shape
    reasoning_mask = torch.ones(B, T, dtype=torch.bool, device=input_ids.device)

    for b in range(B):
        # Find first occurrence of answer_start_token_id
        answer_positions = (input_ids[b] == answer_start_token_id).nonzero(as_tuple=False)
        if answer_positions.numel() > 0:
            answer_start = answer_positions[0].item()
            # Mark everything from answer_start onward as NOT reasoning
            reasoning_mask[b, answer_start:] = False

        # Also mask padding
        pad_positions = (input_ids[b] == pad_token_id).nonzero(as_tuple=False)
        if pad_positions.numel() > 0:
            first_pad = pad_positions[0].item()
            reasoning_mask[b, first_pad:] = False

    return reasoning_mask  # (B, T) bool


def batch_verify(
    problems: List[str],
    generated_answers: List[str],
    ground_truths: List[str],
    task_type: str,  # "math", "gsm8k", or "bbh"
    majority_vote_samples: Optional[List[List[str]]] = None,  # required for bbh
    timeout_seconds: int = 10,
) -> List[bool]:
    """
    §4.2 / §5.1  Batch verification dispatcher.

    Routes each problem to the appropriate verifier based on task_type.

    Args:
        problems:               list of problem strings
        generated_answers:      list of model-generated answer strings
        ground_truths:          list of ground truth strings
        task_type:              "math" | "gsm8k" | "bbh"
        majority_vote_samples:  for BBH only — list of sample lists per problem
        timeout_seconds:        interpreter timeout

    Returns:
        results: list of bool — True if verified correct
    """
    results = []
    for i, (prob, gen, gt) in enumerate(zip(problems, generated_answers, ground_truths)):
        if task_type in ("math", "gsm8k"):
            correct = verify_math_python_interpreter(prob, gen, gt, timeout_seconds)
        elif task_type == "bbh":
            samples = majority_vote_samples[i] if majority_vote_samples else [gen]
            correct = verify_bbh_majority_voting(prob, samples, gt)
        else:
            raise ValueError(f"Unknown task_type '{task_type}'. Use 'math', 'gsm8k', or 'bbh'.")
        results.append(correct)

    return results
