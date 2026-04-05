"""
loss.py — VGEE Loss Functions
Verification-Gated Epistemic Exploration (§4.3, Eq 3–5)

Implements:
  - Conditional KL-Regularizer with Cases A / B / C  (§4.3)
  - VGEE PPO objective  (Eq 5)
  - PPO clip loss  (standard PPO, referenced by paper)
  - Value function loss  (Eq 5, L^VF term)

Paper notation preserved throughout:
  β_base  — base KL penalty coefficient
  β_eff   — effective KL penalty (case-conditional)
  κ       — strict KL multiplier for Case B (κ >> 1)
  δ       — uncertainty threshold
  U_τ     — trajectory uncertainty (Eq 2)
  c_1     — value function coefficient (Eq 5)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# §4.3  Conditional KL-Regularizer Case Labels
# ---------------------------------------------------------------------------

# Case identifiers per §4.3
CASE_A_DISCOVERY = "discovery"          # high uncertainty + verified correct
CASE_B_FAILED_EXPLORATION = "failed"    # high uncertainty + verification failed
CASE_C_EXPLOITATION = "exploitation"    # low uncertainty


# ---------------------------------------------------------------------------
# §4.3  Conditional KL Case Assignment
# ---------------------------------------------------------------------------

def assign_kl_cases(
    high_uncertainty_mask: torch.Tensor,  # (B,) bool — U_τ >= δ
    verified_correct: torch.Tensor,        # (B,) bool — verification result
                                           # (False for low-uncertainty trajectories)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    §4.3 — Assign each trajectory to Case A, B, or C.

    Case A (Discovery):         U_τ >= δ AND verified correct
    Case B (Failed Exploration):U_τ >= δ AND NOT verified correct
    Case C (Exploitation):      U_τ < δ  (verification not triggered)

    Args:
        high_uncertainty_mask: (B,) bool — True where U_τ >= δ
        verified_correct:      (B,) bool — True where verifier returned correct
                                           For Case C trajectories this should be False
                                           (they were not sent to verifier)

    Returns:
        case_a_mask: (B,) bool — trajectories in Case A (Discovery)
        case_b_mask: (B,) bool — trajectories in Case B (Failed Exploration)
        case_c_mask: (B,) bool — trajectories in Case C (Exploitation)
    """
    # Case A: §4.3 "high uncertainty AND verified correct" [PAPER]
    case_a_mask = high_uncertainty_mask & verified_correct   # (B,) bool

    # Case B: §4.3 "high uncertainty AND verification failed" [PAPER]
    case_b_mask = high_uncertainty_mask & ~verified_correct  # (B,) bool

    # Case C: §4.3 "low uncertainty — standard exploitation" [PAPER]
    case_c_mask = ~high_uncertainty_mask                     # (B,) bool

    return case_a_mask, case_b_mask, case_c_mask


def compute_effective_beta(
    high_uncertainty_mask: torch.Tensor,  # (B,) bool
    verified_correct: torch.Tensor,        # (B,) bool
    beta_base: float,                      # β_base [UNSPECIFIED — see configs/base.yaml]
    kappa: float,                          # κ >> 1 [UNSPECIFIED — paper says κ >> 1]
) -> torch.Tensor:
    """
    §4.3 — Compute per-trajectory effective KL penalty coefficient β_eff.

    Eq 3 (Case A — Discovery):
        β_eff = β_base

    Eq 4 (Case B — Failed Exploration):
        β_eff = κ · β_base    where κ >> 1

    Case C (Exploitation): β_eff = β_base  (standard penalty, paper implies this)
    [UNSPECIFIED] Paper does not explicitly state β_eff for Case C.
    Using β_base is the natural default (standard PPO KL).

    Args:
        high_uncertainty_mask: (B,) bool — U_τ >= δ
        verified_correct:      (B,) bool — verifier output
        beta_base:             float — β_base
        kappa:                 float — κ (must be >> 1)

    Returns:
        beta_eff: (B,) float — per-trajectory effective KL coefficient
    """
    B = high_uncertainty_mask.shape[0]
    device = high_uncertainty_mask.device

    case_a_mask, case_b_mask, case_c_mask = assign_kl_cases(
        high_uncertainty_mask, verified_correct
    )

    # Initialize all to β_base (covers Cases A and C)
    beta_eff = torch.full((B,), beta_base, dtype=torch.float32, device=device)  # (B,)

    # Eq 4: Case B — β_eff = κ · β_base
    beta_eff = torch.where(
        case_b_mask,
        torch.full_like(beta_eff, kappa * beta_base),
        beta_eff,
    )  # (B,)

    return beta_eff  # (B,)


# ---------------------------------------------------------------------------
# §4.3  KL Divergence (token-averaged)
# ---------------------------------------------------------------------------

def compute_per_trajectory_kl(
    policy_log_probs: torch.Tensor,    # (B, T, V) — log p_θ(v | context)
    ref_log_probs: torch.Tensor,       # (B, T, V) — log p_ref(v | context)
    attention_mask: torch.Tensor,      # (B, T)    — 1=valid, 0=pad
) -> torch.Tensor:
    """
    §4.3 — Per-trajectory KL divergence: KL(π_θ || π_ref).

    KL(π_θ || π_ref) = Σ_t Σ_v p_θ(v|t) [log p_θ(v|t) - log p_ref(v|t)]
                     = Σ_t KL_t

    We return the mean over valid (non-padded) tokens for stability.

    [UNSPECIFIED] Paper writes "Conditional_KL(θ)" in Eq 5 without specifying
    whether it is token-summed or token-averaged. Token-averaged is standard
    in RLVR work (e.g., DeepSeekMath, DAPO).

    Args:
        policy_log_probs: (B, T, V) — log probs under policy
        ref_log_probs:    (B, T, V) — log probs under reference
        attention_mask:   (B, T)    — padding mask

    Returns:
        kl_per_trajectory: (B,) — mean per-token KL for each trajectory
    """
    # Recover policy probabilities from log-probs
    policy_probs = policy_log_probs.exp()  # (B, T, V)

    # Per-token KL: Σ_v p(v) [log p(v) - log q(v)]
    per_token_kl = (policy_probs * (policy_log_probs - ref_log_probs)).sum(dim=-1)  # (B, T)

    # Mask padding positions
    per_token_kl = per_token_kl * attention_mask.float()  # (B, T)

    # Average over valid token positions
    token_counts = attention_mask.float().sum(dim=-1).clamp(min=1)  # (B,)
    kl_per_trajectory = per_token_kl.sum(dim=-1) / token_counts     # (B,)

    return kl_per_trajectory  # (B,)


def conditional_kl_loss(
    policy_log_probs: torch.Tensor,    # (B, T, V)
    ref_log_probs: torch.Tensor,       # (B, T, V)
    attention_mask: torch.Tensor,      # (B, T)
    high_uncertainty_mask: torch.Tensor,  # (B,) bool
    verified_correct: torch.Tensor,       # (B,) bool
    beta_base: float,
    kappa: float,
) -> Tuple[torch.Tensor, dict]:
    """
    §4.3 — Conditional KL-Regularizer.

    Conditional_KL(θ) = Σ_i β_eff(i) · KL(π_θ(·|x_i) || π_ref(·|x_i))

    where β_eff depends on the case assignment (Eqs 3–4):
      Case A (Discovery):          β_eff = β_base
      Case B (Failed Exploration): β_eff = κ · β_base
      Case C (Exploitation):       β_eff = β_base  [UNSPECIFIED — implied default]

    Args:
        policy_log_probs:     (B, T, V) — log p_θ
        ref_log_probs:        (B, T, V) — log p_ref
        attention_mask:       (B, T)
        high_uncertainty_mask:(B,) bool — U_τ >= δ
        verified_correct:     (B,) bool — verifier output
        beta_base:            float — β_base
        kappa:                float — κ (Case B multiplier, κ >> 1)

    Returns:
        loss:   scalar — mean conditional KL loss
        info:   dict   — diagnostic info (case counts, mean KL per case)
    """
    # Step 1: compute raw KL per trajectory
    kl_per_traj = compute_per_trajectory_kl(
        policy_log_probs, ref_log_probs, attention_mask
    )  # (B,)

    # Step 2: compute effective β per trajectory
    beta_eff = compute_effective_beta(
        high_uncertainty_mask, verified_correct, beta_base, kappa
    )  # (B,)

    # Step 3: weighted KL
    weighted_kl = beta_eff * kl_per_traj  # (B,)

    # Step 4: mean over batch
    loss = weighted_kl.mean()  # scalar

    # Diagnostics
    case_a_mask, case_b_mask, case_c_mask = assign_kl_cases(
        high_uncertainty_mask, verified_correct
    )

    info = {
        "conditional_kl_loss": loss.item(),
        "n_case_a_discovery": case_a_mask.sum().item(),
        "n_case_b_failed": case_b_mask.sum().item(),
        "n_case_c_exploitation": case_c_mask.sum().item(),
        "mean_kl_case_a": kl_per_traj[case_a_mask].mean().item() if case_a_mask.any() else 0.0,
        "mean_kl_case_b": kl_per_traj[case_b_mask].mean().item() if case_b_mask.any() else 0.0,
        "mean_kl_case_c": kl_per_traj[case_c_mask].mean().item() if case_c_mask.any() else 0.0,
        "mean_beta_eff": beta_eff.mean().item(),
    }

    return loss, info


# ---------------------------------------------------------------------------
# PPO Components  [PAPER §5.1, Eq 5]
# ---------------------------------------------------------------------------

def ppo_clip_loss(
    log_probs_new: torch.Tensor,   # (B, T) — log π_θ(a_t | s_t) under current policy
    log_probs_old: torch.Tensor,   # (B, T) — log π_θ_old(a_t | s_t) from rollout
    advantages: torch.Tensor,      # (B, T) — GAE advantages
    attention_mask: torch.Tensor,  # (B, T) — 1=valid, 0=pad
    clip_epsilon: float = 0.2,     # ε_clip [UNSPECIFIED — standard PPO default 0.2]
) -> Tuple[torch.Tensor, dict]:
    """
    Standard PPO clipped surrogate objective L^CLIP.

    Eq 5: L(θ) = E_t [L^CLIP(θ) - c_1 L^VF(θ) + Conditional_KL(θ)]

    L^CLIP(θ) = E_t [min(r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t)]

    where r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

    Paper references PPO [PAPER §5.1] but does not state clip_epsilon.
    [UNSPECIFIED] clip_epsilon=0.2 is the canonical default from Schulman et al. 2017.
    Alternatives: 0.1, 0.3.

    Args:
        log_probs_new:   (B, T) — log probs under current (updating) policy
        log_probs_old:   (B, T) — log probs from rollout (fixed)
        advantages:      (B, T) — GAE-normalized advantages
        attention_mask:  (B, T) — padding mask
        clip_epsilon:    float  — PPO clip parameter ε [UNSPECIFIED]

    Returns:
        clip_loss: scalar — negated mean clipped objective (minimized)
        info:      dict   — diagnostics
    """
    # Importance sampling ratio r_t = π_θ / π_θ_old
    log_ratio = log_probs_new - log_probs_old  # (B, T)
    ratio = log_ratio.exp()                    # (B, T) — r_t(θ)

    # Clipped surrogate
    surr1 = ratio * advantages                                              # (B, T)
    surr2 = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages  # (B, T)

    # L^CLIP = E[min(surr1, surr2)]  — we negate for minimization
    clip_obj = torch.min(surr1, surr2)  # (B, T)

    # Mask padding
    clip_obj = clip_obj * attention_mask.float()  # (B, T)

    # Average over valid tokens
    token_count = attention_mask.float().sum().clamp(min=1)
    clip_loss = -clip_obj.sum() / token_count  # scalar (negated: we minimize)

    # Diagnostics
    with torch.no_grad():
        approx_kl = (log_ratio * attention_mask.float()).sum() / token_count
        clip_fraction = (
            ((ratio - 1.0).abs() > clip_epsilon).float() * attention_mask.float()
        ).sum() / token_count

    info = {
        "ppo_clip_loss": clip_loss.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": clip_fraction.item(),
        "mean_ratio": ratio[attention_mask.bool()].mean().item(),
    }

    return clip_loss, info


def value_function_loss(
    values_pred: torch.Tensor,     # (B, T) — predicted values V(s_t)
    values_old: torch.Tensor,      # (B, T) — values from rollout (for clip)
    returns: torch.Tensor,         # (B, T) — GAE returns (targets for value fn)
    attention_mask: torch.Tensor,  # (B, T)
    clip_epsilon: float = 0.2,     # value clip [UNSPECIFIED — same as PPO clip]
) -> Tuple[torch.Tensor, dict]:
    """
    PPO value function loss L^VF (Eq 5 term c_1 L^VF).

    L^VF = E_t [max((V_θ(s_t) - R_t)^2, (clip(V_θ, V_old-ε, V_old+ε) - R_t)^2)]

    [UNSPECIFIED] Paper includes c_1 L^VF in Eq 5 but does not detail value
    function architecture. We assume a shared value head on the LM backbone,
    common in RLVR implementations (e.g., TRL, OpenRLHF).

    [UNSPECIFIED] clip_epsilon for value function loss — using same as policy.
    Alternatives: 0.2 (PPO default), 0.1, None (no clipping).

    Args:
        values_pred:    (B, T) — current value predictions
        values_old:     (B, T) — rollout value predictions
        returns:        (B, T) — discounted returns
        attention_mask: (B, T)
        clip_epsilon:   float  — value clip parameter [UNSPECIFIED]

    Returns:
        vf_loss: scalar
        info:    dict
    """
    # Unclipped value loss
    vf_loss1 = (values_pred - returns).pow(2)  # (B, T)

    # Clipped value loss
    values_clipped = values_old + (values_pred - values_old).clamp(
        -clip_epsilon, clip_epsilon
    )  # (B, T)
    vf_loss2 = (values_clipped - returns).pow(2)  # (B, T)

    # Take max (conservative)
    vf_loss_per_token = torch.max(vf_loss1, vf_loss2)  # (B, T)

    # Mask and average
    vf_loss_per_token = vf_loss_per_token * attention_mask.float()  # (B, T)
    token_count = attention_mask.float().sum().clamp(min=1)
    vf_loss = 0.5 * vf_loss_per_token.sum() / token_count  # scalar

    info = {
        "vf_loss": vf_loss.item(),
        "mean_value_pred": values_pred[attention_mask.bool()].mean().item(),
        "mean_return": returns[attention_mask.bool()].mean().item(),
    }

    return vf_loss, info


# ---------------------------------------------------------------------------
# §4.3  VGEE Full Objective  (Eq 5)
# ---------------------------------------------------------------------------

@dataclass
class VGEELossOutput:
    """Container for the full VGEE loss breakdown."""
    total_loss: torch.Tensor          # Eq 5 full objective
    clip_loss: torch.Tensor           # L^CLIP
    vf_loss: torch.Tensor             # L^VF
    conditional_kl_loss: torch.Tensor # Conditional_KL(θ)
    info: dict                        # all diagnostics merged


def vgee_loss(
    # Policy log-probs (current update step)
    policy_log_probs_token: torch.Tensor,   # (B, T) — log π_θ(a_t|s_t) per token
    policy_log_probs_full: torch.Tensor,    # (B, T, V) — full distribution for KL
    # Reference model log-probs
    ref_log_probs_full: torch.Tensor,       # (B, T, V) — log π_ref distribution
    # Old policy (from rollout)
    old_log_probs_token: torch.Tensor,      # (B, T) — log π_θ_old(a_t|s_t)
    # Value function
    values_pred: torch.Tensor,              # (B, T) — V_θ(s_t)
    values_old: torch.Tensor,               # (B, T) — V_θ_old(s_t) from rollout
    # Returns and advantages
    advantages: torch.Tensor,               # (B, T) — GAE advantages
    returns: torch.Tensor,                  # (B, T) — GAE returns
    # Masks
    attention_mask: torch.Tensor,           # (B, T)
    # Verification gate results
    high_uncertainty_mask: torch.Tensor,    # (B,) bool — U_τ >= δ
    verified_correct: torch.Tensor,         # (B,) bool — verifier output
    # Hyperparameters from config
    beta_base: float,                       # β_base [UNSPECIFIED]
    kappa: float,                           # κ >> 1 [UNSPECIFIED]
    c_1: float,                             # value function coefficient [UNSPECIFIED]
    clip_epsilon: float,                    # PPO clip ε [UNSPECIFIED]
) -> VGEELossOutput:
    """
    Eq 5 — VGEE full training objective:

        L(θ) = E_t [L^CLIP(θ) - c_1 · L^VF(θ) + Conditional_KL(θ)]

    All three terms are computed and combined here.

    Args:
        policy_log_probs_token: (B, T)    — log π_θ(a_t|s_t) for clip loss
        policy_log_probs_full:  (B, T, V) — full distribution for KL
        ref_log_probs_full:     (B, T, V) — reference distribution for KL
        old_log_probs_token:    (B, T)    — old policy log probs for ratio
        values_pred:            (B, T)    — current value predictions
        values_old:             (B, T)    — rollout value predictions
        advantages:             (B, T)    — GAE advantages
        returns:                (B, T)    — GAE returns
        attention_mask:         (B, T)
        high_uncertainty_mask:  (B,) bool
        verified_correct:       (B,) bool
        beta_base:              float
        kappa:                  float
        c_1:                    float — [UNSPECIFIED] value fn coefficient
        clip_epsilon:           float — [UNSPECIFIED] PPO clip ε

    Returns:
        VGEELossOutput with total_loss and per-term breakdown
    """
    # Term 1: L^CLIP(θ) [PAPER Eq 5]
    l_clip, clip_info = ppo_clip_loss(
        log_probs_new=policy_log_probs_token,
        log_probs_old=old_log_probs_token,
        advantages=advantages,
        attention_mask=attention_mask,
        clip_epsilon=clip_epsilon,
    )  # scalar

    # Term 2: c_1 · L^VF(θ) [PAPER Eq 5]
    l_vf, vf_info = value_function_loss(
        values_pred=values_pred,
        values_old=values_old,
        returns=returns,
        attention_mask=attention_mask,
    )  # scalar

    # Term 3: Conditional_KL(θ) [PAPER §4.3, Eq 5]
    l_kl, kl_info = conditional_kl_loss(
        policy_log_probs=policy_log_probs_full,
        ref_log_probs=ref_log_probs_full,
        attention_mask=attention_mask,
        high_uncertainty_mask=high_uncertainty_mask,
        verified_correct=verified_correct,
        beta_base=beta_base,
        kappa=kappa,
    )  # scalar

    # Eq 5: L(θ) = L^CLIP - c_1 · L^VF + Conditional_KL
    # Note: L^CLIP is already negated (we minimize), so:
    # total = l_clip + c_1 * l_vf + l_kl
    total_loss = l_clip + c_1 * l_vf + l_kl  # scalar

    info = {
        "total_loss": total_loss.item(),
        **clip_info,
        **vf_info,
        **kl_info,
    }

    return VGEELossOutput(
        total_loss=total_loss,
        clip_loss=l_clip,
        vf_loss=l_vf,
        conditional_kl_loss=l_kl,
        info=info,
    )


# ---------------------------------------------------------------------------
# GAE (Generalized Advantage Estimation)  [PAPER §5.1: γ=0.99, λ=0.95]
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: torch.Tensor,    # (B, T) — per-token rewards
    values: torch.Tensor,     # (B, T) — value estimates V(s_t)
    attention_mask: torch.Tensor,  # (B, T) — 1=valid
    gamma: float = 0.99,      # discount factor γ [PAPER §5.1]
    lam: float = 0.95,        # GAE lambda λ [PAPER §5.1]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generalized Advantage Estimation (GAE).

    Paper §5.1: "We use PPO with GAE, γ=0.99, λ=0.95."

    A_t^GAE = Σ_{l=0}^{T-t} (γλ)^l δ_{t+l}

    where δ_t = r_t + γ V(s_{t+1}) - V(s_t)

    Args:
        rewards:        (B, T) — r_t (typically sparse: 0 except at final token)
        values:         (B, T) — V_θ(s_t) from value head
        attention_mask: (B, T) — padding mask
        gamma:          float  — γ=0.99 [PAPER]
        lam:            float  — λ=0.95 [PAPER]

    Returns:
        advantages: (B, T) — GAE advantages (normalized)
        returns:    (B, T) — discounted returns = advantages + values
    """
    B, T = rewards.shape
    advantages = torch.zeros_like(rewards)  # (B, T)

    # Next value: 0 at terminal positions
    next_value = torch.zeros(B, device=rewards.device)  # (B,)

    # Backward pass through time
    gae = torch.zeros(B, device=rewards.device)  # (B,)
    for t in reversed(range(T)):
        # Mask: 1 if this position is valid
        mask_t = attention_mask[:, t].float()  # (B,)

        # Bootstrap value at t+1
        # [UNSPECIFIED] Terminal handling: if last valid token, next_value = 0
        next_val = next_value  # (B,)

        # TD error: δ_t = r_t + γ V(s_{t+1}) - V(s_t)
        delta = rewards[:, t] + gamma * next_val - values[:, t]  # (B,)

        # GAE: A_t = δ_t + γλ A_{t+1}
        gae = delta + gamma * lam * gae  # (B,)

        # Zero-out padded positions
        gae = gae * mask_t  # (B,)

        advantages[:, t] = gae  # (B,)
        next_value = values[:, t] * mask_t  # (B,) — becomes next step's bootstrap

    # Returns = advantages + values (baseline)
    returns = advantages + values  # (B, T)

    # Normalize advantages [UNSPECIFIED — standard practice, not stated in paper]
    # Alternatives: per-batch norm, no norm, running mean/std
    valid_adv = advantages[attention_mask.bool()]
    if valid_adv.numel() > 1:
        adv_mean = valid_adv.mean()
        adv_std = valid_adv.std().clamp(min=1e-8)
        advantages = (advantages - adv_mean) / adv_std  # normalize [UNSPECIFIED]
        # Re-zero padded positions after normalization
        advantages = advantages * attention_mask.float()

    return advantages, returns  # both (B, T)
