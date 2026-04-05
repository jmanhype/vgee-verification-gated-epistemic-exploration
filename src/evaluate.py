"""
evaluate.py — VGEE Evaluation Metrics
Verification-Gated Epistemic Exploration (§5.2)

Implements:
  - Accuracy metric  [PAPER §5.2]
  - ECE (Expected Calibration Error)  [PAPER §5.2]
  - Reasoning diversity metric  [PAPER §5.2]

Paper §5.2: "We evaluate on MATH500, GSM8K, and BBH using accuracy, ECE,
and reasoning diversity."

All UNSPECIFIED details flagged inline.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data import ReasoningDataset, TASK_VERIFIER, create_dataloader
from .utils import batch_verify

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# §5.2  Accuracy
# ---------------------------------------------------------------------------

def compute_accuracy(
    predictions: List[str],     # model-generated answers
    ground_truths: List[str],   # correct answers
    task_type: str,             # "math", "gsm8k", or "bbh"
    majority_vote_samples: Optional[List[List[str]]] = None,
) -> float:
    """
    §5.2 — Task accuracy metric.

    Paper: "We report accuracy on MATH500, GSM8K, and BBH."

    For MATH/GSM8K: verification via Python interpreter.
    For BBH: majority vote among multiple samples.

    [UNSPECIFIED] Whether the paper uses greedy decoding or sampling for
    evaluation. We use greedy (temperature=0 / do_sample=False) by default.
    Alternatives: pass@k, majority@k.

    Args:
        predictions:            list of generated answer strings
        ground_truths:          list of correct answer strings
        task_type:              "math" | "gsm8k" | "bbh"
        majority_vote_samples:  for BBH, extra sample lists per problem

    Returns:
        accuracy: float in [0, 1]
    """
    if not predictions:
        return 0.0

    correctness = batch_verify(
        problems=[""] * len(predictions),  # problem text not needed for accuracy
        generated_answers=predictions,
        ground_truths=ground_truths,
        task_type=task_type,
        majority_vote_samples=majority_vote_samples,
    )

    return sum(correctness) / len(correctness)


# ---------------------------------------------------------------------------
# §5.2  ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------

def compute_ece(
    confidences: torch.Tensor,  # (N,) — model confidence for each prediction
    correctness: torch.Tensor,  # (N,) bool — whether each prediction is correct
    n_bins: int = 15,           # number of calibration bins [UNSPECIFIED — standard is 15]
) -> float:
    """
    §5.2 — Expected Calibration Error (ECE).

    Paper: "We report ECE as a calibration metric."

    ECE = Σ_{b=1}^{B} (|B_b| / N) |acc(B_b) - conf(B_b)|

    where B_b is the set of samples in bin b,
          acc(B_b) is the mean accuracy in the bin,
          conf(B_b) is the mean confidence in the bin.

    [UNSPECIFIED] Definition of "confidence" for generative models.
    We use the mean probability assigned to the generated answer tokens
    as a proxy for model confidence.

    [UNSPECIFIED] n_bins: standard ECE uses 15 equal-width bins.

    Args:
        confidences: (N,) — confidence scores in [0, 1]
        correctness: (N,) bool — whether each sample was correct
        n_bins:      int — number of equal-width calibration bins

    Returns:
        ece: float — expected calibration error
    """
    N = len(confidences)
    if N == 0:
        return 0.0

    confidences = confidences.float().clamp(0.0, 1.0)  # (N,)
    correctness = correctness.float()                   # (N,)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)  # (n_bins+1,)
    ece = 0.0

    for i in range(n_bins):
        lower = bin_boundaries[i].item()
        upper = bin_boundaries[i + 1].item()

        # Include upper bound in last bin for samples with confidence == 1.0
        if i < n_bins - 1:
            in_bin = (confidences >= lower) & (confidences < upper)
        else:
            in_bin = (confidences >= lower) & (confidences <= upper)

        n_in_bin = in_bin.sum().item()
        if n_in_bin == 0:
            continue

        bin_acc = correctness[in_bin].mean().item()
        bin_conf = confidences[in_bin].mean().item()

        ece += (n_in_bin / N) * abs(bin_acc - bin_conf)

    return ece


def extract_confidence_from_log_probs(
    token_log_probs: torch.Tensor,  # (N, gen_len) — log-probs of generated tokens
    attention_mask: torch.Tensor,   # (N, gen_len) — 1=valid
) -> torch.Tensor:
    """
    Derive a scalar confidence from per-token log-probabilities.

    [UNSPECIFIED] Paper does not specify how confidence is derived for LLMs.

    Approach: mean per-token probability over generated (non-pad) tokens,
    then average as a sequence-level confidence score.

    Alternatives:
      - geometric mean of probabilities
      - probability of the final answer token only
      - min probability (most conservative)

    Args:
        token_log_probs: (N, T) — log-probs of generated tokens
        attention_mask:  (N, T) — padding mask

    Returns:
        confidences: (N,) — scalar confidence per sample in [0, 1]
    """
    # Mean token probability
    probs = token_log_probs.exp()                          # (N, T) in [0, 1]
    masked_probs = probs * attention_mask.float()          # (N, T)
    token_counts = attention_mask.float().sum(dim=-1).clamp(min=1)  # (N,)
    mean_prob = masked_probs.sum(dim=-1) / token_counts    # (N,)

    return mean_prob.clamp(0.0, 1.0)  # (N,)


# ---------------------------------------------------------------------------
# §5.2  Reasoning Diversity
# ---------------------------------------------------------------------------

def compute_reasoning_diversity(
    trajectories_per_prompt: List[List[str]],   # list of lists: [[traj1, traj2, ...], ...]
    n_samples: int = 8,                          # K samples per prompt [UNSPECIFIED]
) -> float:
    """
    §5.2 — Reasoning diversity metric.

    Paper: "We measure reasoning diversity to evaluate whether VGEE encourages
    diverse exploration rather than mode collapse."

    [UNSPECIFIED] Paper mentions "reasoning diversity" as a metric but does not
    define the exact computation. Common approaches:

      1. Distinct n-gram diversity (Self-BLEU inverse)
      2. Unique solution paths per prompt
      3. Average pairwise edit distance
      4. Vocabulary diversity (Type-Token Ratio)

    We implement pairwise n-gram diversity (Self-BLEU-4 inverse), which is
    standard in text diversity literature.

    [UNSPECIFIED] Alternatives: edit distance, embedding cosine dissimilarity.

    Args:
        trajectories_per_prompt: list of lists of trajectory strings
                                 trajectories_per_prompt[i] = K trajectories for prompt i
        n_samples:               expected K per prompt [UNSPECIFIED]

    Returns:
        diversity_score: float — mean diversity across prompts (higher = more diverse)
    """
    if not trajectories_per_prompt:
        return 0.0

    diversity_scores = []
    for trajectories in trajectories_per_prompt:
        if len(trajectories) < 2:
            diversity_scores.append(0.0)
            continue

        # Compute pairwise n-gram overlap as diversity proxy
        pairwise_sims = []
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                sim = _ngram_similarity(trajectories[i], trajectories[j], n=4)
                pairwise_sims.append(sim)

        # Diversity = 1 - mean pairwise similarity
        mean_sim = sum(pairwise_sims) / len(pairwise_sims)
        diversity_scores.append(1.0 - mean_sim)

    return sum(diversity_scores) / len(diversity_scores)


def _ngram_similarity(text_a: str, text_b: str, n: int = 4) -> float:
    """
    Compute n-gram overlap (Jaccard similarity) between two strings.

    [UNSPECIFIED] Tokenization: word-level n-grams.

    Args:
        text_a: first string
        text_b: second string
        n:      n-gram order

    Returns:
        similarity: float in [0, 1]
    """
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()

    def get_ngrams(tokens, n):
        return set(tuple(tokens[i: i + n]) for i in range(max(0, len(tokens) - n + 1)))

    ngrams_a = get_ngrams(tokens_a, n)
    ngrams_b = get_ngrams(tokens_b, n)

    if not ngrams_a and not ngrams_b:
        return 1.0  # both empty — identical
    if not ngrams_a or not ngrams_b:
        return 0.0  # one empty — no overlap

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b

    return len(intersection) / len(union)  # Jaccard similarity


def compute_unique_solution_rate(
    trajectories_per_prompt: List[List[str]],
) -> float:
    """
    Alternative diversity metric: fraction of unique final answers per prompt.

    [UNSPECIFIED] Paper does not specify this metric; provided as alternative.

    Returns:
        unique_rate: mean fraction of unique answers across prompts
    """
    from .utils import _extract_final_answer
    rates = []
    for trajectories in trajectories_per_prompt:
        if not trajectories:
            continue
        answers = [_extract_final_answer(t).strip().lower() for t in trajectories]
        unique_fraction = len(set(answers)) / len(answers)
        rates.append(unique_fraction)
    return sum(rates) / len(rates) if rates else 0.0


# ---------------------------------------------------------------------------
# §5.2  Full Evaluation Loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_task(
    model_with_vh,                         # VGEEModelWithValueHead
    eval_dataset: ReasoningDataset,
    cfg: dict,
    device: torch.device,
    n_diversity_samples: int = 8,          # K for diversity [UNSPECIFIED]
    greedy_decode: bool = True,            # [UNSPECIFIED] paper doesn't specify
) -> Dict:
    """
    §5.2 — Evaluate VGEE on a single task dataset.

    Computes:
      - Accuracy
      - ECE
      - Reasoning diversity

    [UNSPECIFIED] Whether paper uses greedy or sampled decoding for evaluation.
    Greedy (do_sample=False) is most common for final evaluation.

    Args:
        model_with_vh:       VGEEModelWithValueHead
        eval_dataset:        evaluation ReasoningDataset
        cfg:                 config dict
        device:              torch device
        n_diversity_samples: K samples for diversity metric [UNSPECIFIED]
        greedy_decode:       use greedy decoding [UNSPECIFIED]

    Returns:
        metrics dict: {accuracy, ece, reasoning_diversity}
    """
    from transformers import GenerationConfig
    from .data import collate_reasoning_batch

    model_with_vh.eval()
    tokenizer = model_with_vh.vgee.tokenizer

    eval_loader = create_dataloader(
        dataset=eval_dataset,
        tokenizer=tokenizer,
        batch_size=32,          # [UNSPECIFIED] eval batch size
        shuffle=False,
        num_workers=0,
    )

    all_predictions = []
    all_ground_truths = []
    all_task_types = []
    all_confidences = []
    all_trajectories_per_prompt = []

    # Greedy generation config
    greedy_config = GenerationConfig(
        max_new_tokens=cfg["training"]["max_new_tokens"],
        do_sample=False,          # greedy [UNSPECIFIED]
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Sampling config for diversity evaluation
    sampling_config = GenerationConfig(
        max_new_tokens=cfg["training"]["max_new_tokens"],
        do_sample=True,
        temperature=1.0,          # [UNSPECIFIED]
        top_p=0.9,                # [UNSPECIFIED]
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    for batch in eval_loader:
        input_ids = batch["input_ids"].to(device)         # (B, P)
        attention_mask = batch["attention_mask"].to(device)
        ground_truths = batch["answers"]
        task_types = batch["task_types"]
        B, P = input_ids.shape

        # ----------------------------------------------------------------
        # Greedy decode for accuracy
        # ----------------------------------------------------------------
        gen = model_with_vh.vgee.policy_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=greedy_config,
            return_dict_in_generate=True,
            output_scores=True,
        )

        full_ids = gen.sequences                   # (B, full_len)
        gen_ids = full_ids[:, P:]                  # (B, gen_len)

        # Decode predictions
        predictions = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        # Compute per-token log-probs for confidence
        if gen.scores:
            stacked_logits = torch.stack(gen.scores, dim=1)  # (B, gen_len, V)
            log_probs = F.log_softmax(stacked_logits, dim=-1)  # (B, gen_len, V)
            # Log-prob of selected tokens
            token_lp = log_probs.gather(
                dim=-1,
                index=gen_ids.unsqueeze(-1)
            ).squeeze(-1)  # (B, gen_len)
            gen_mask = (gen_ids != tokenizer.pad_token_id).long()  # (B, gen_len)
            confidences = extract_confidence_from_log_probs(token_lp, gen_mask)  # (B,)
        else:
            # Fallback: uniform confidence [UNSPECIFIED]
            confidences = torch.full((B,), 0.5, device=device)

        all_predictions.extend(predictions)
        all_ground_truths.extend(ground_truths)
        all_task_types.extend(task_types)
        all_confidences.append(confidences.cpu())

        # ----------------------------------------------------------------
        # Sample K trajectories for diversity (§5.2)
        # ----------------------------------------------------------------
        # [UNSPECIFIED] Paper mentions diversity but not how many samples
        traj_texts_batch = [[] for _ in range(B)]
        for _ in range(n_diversity_samples):
            sampled_gen = model_with_vh.vgee.policy_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=sampling_config,
                return_dict_in_generate=False,
            )  # (B, full_len)
            sampled_texts = tokenizer.batch_decode(
                sampled_gen[:, P:],
                skip_special_tokens=True,
            )
            for b in range(B):
                traj_texts_batch[b].append(sampled_texts[b])

        all_trajectories_per_prompt.extend(traj_texts_batch)

    # ----------------------------------------------------------------
    # Compute metrics
    # ----------------------------------------------------------------
    # Accuracy
    task_type = all_task_types[0] if all_task_types else "math"
    accuracy = compute_accuracy(
        predictions=all_predictions,
        ground_truths=all_ground_truths,
        task_type=task_type,
    )

    # ECE
    all_conf_tensor = torch.cat(all_confidences, dim=0)  # (N,)
    correctness_list = batch_verify(
        problems=[""] * len(all_predictions),
        generated_answers=all_predictions,
        ground_truths=all_ground_truths,
        task_type=task_type,
    )
    correctness_tensor = torch.tensor(correctness_list, dtype=torch.bool)
    ece = compute_ece(
        confidences=all_conf_tensor,
        correctness=correctness_tensor,
        n_bins=cfg["evaluation"]["ece_n_bins"],
    )

    # Reasoning diversity
    diversity = compute_reasoning_diversity(
        trajectories_per_prompt=all_trajectories_per_prompt,
        n_samples=n_diversity_samples,
    )

    return {
        "accuracy": accuracy,
        "ece": ece,
        "reasoning_diversity": diversity,
        "n_samples": len(all_predictions),
    }


def evaluate_vgee(
    model_with_vh,
    eval_datasets: Dict[str, ReasoningDataset],
    cfg: dict,
    device: torch.device,
) -> Dict[str, Dict]:
    """
    §5.2 — Run evaluation across all eval datasets.

    Paper §5.2: "We evaluate on MATH500, GSM8K, and BBH."

    Args:
        model_with_vh:  VGEEModelWithValueHead
        eval_datasets:  dict of task_name → ReasoningDataset
        cfg:            config dict
        device:         torch device

    Returns:
        results: dict of task_name → {accuracy, ece, reasoning_diversity}
    """
    results = {}
    for task_name, dataset in eval_datasets.items():
        logger.info(f"Evaluating {task_name}...")
        metrics = evaluate_task(
            model_with_vh=model_with_vh,
            eval_dataset=dataset,
            cfg=cfg,
            device=device,
            n_diversity_samples=cfg["evaluation"]["diversity_samples"],
        )
        results[task_name] = metrics
        logger.info(
            f"  {task_name}: acc={metrics['accuracy']:.4f}, "
            f"ece={metrics['ece']:.4f}, "
            f"diversity={metrics['reasoning_diversity']:.4f}"
        )

    return results
