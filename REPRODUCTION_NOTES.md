# VGEE Reproduction Notes

**Paper**: Verification-Gated Epistemic Exploration: Resolving the Winner-Take-All Paradox in RLVR

This document details every design decision made during implementation, distinguishes paper-specified values from author choices, and flags all gaps that must be resolved before this implementation can reproduce the paper's results.

---

## 1. Paper-Specified Hyperparameters (CONFIRMED)

These values are explicitly stated in the paper and used verbatim:

| Parameter | Value | Paper Location |
|-----------|-------|---------------|
| Training iterations | 10 | §5.1 |
| Batch size | 512 prompts | §5.1 |
| GAE γ | 0.99 | §5.1 |
| GAE λ | 0.95 | §5.1 |
| Learning rate | 1e-5 | §5.1 |
| LR schedule | cosine decay | §5.1 |
| BBH majority vote samples | 5 | §5.1 |
| Base model | decoder-only transformer (e.g., Llama-3-8B) | §5.1 |
| Starting checkpoint | SFT checkpoint | §5.1 |
| U_τ aggregation | max (Eq 2) | Eq 2 |
| KL Case A: β_eff | β_base (Eq 3) | Eq 3 |
| KL Case B: β_eff | κ · β_base (Eq 4) | Eq 4 |
| Verification: MATH/GSM8K | Python interpreter | §5.1 |
| Verification: BBH | Majority voting | §5.1 |
| Eval datasets | MATH500, GSM8K, BBH | §5.2 |
| Eval metrics | Accuracy, ECE, Diversity | §5.2 |

---

## 2. UNSPECIFIED Parameters (Author Choices Made)

These values are **not stated in the paper**. Our defaults are reasonable but may not reproduce exact results. The paper authors must clarify these.

### Critical — likely affects results significantly

| Parameter | Our Default | Paper Says | Alternatives to Try |
|-----------|-------------|------------|---------------------|
| `κ` (Case B KL multiplier) | 10.0 | "κ >> 1" | 5, 20, 50 |
| `β_base` (KL coefficient) | 0.04 | Not stated | 0.01, 0.05, 0.1 |
| `δ` (uncertainty threshold) | 1.0 nat | "tuned on validation" | 0.5, 1.5, 2.0 |
| `K` (trajectories per prompt) | 8 | Not stated | 4, 16 |
| PPO clip ε | 0.2 | Not stated | 0.1, 0.3 |
| Value function coeff `c_1` | 0.5 | Appears in Eq 5, no value | 1.0 |

### Moderate — likely affects training stability

| Parameter | Our Default | Alternatives |
|-----------|-------------|--------------|
| Adam β1 | 0.9 | 0.95 |
| Adam β2 | 0.95 | 0.999 |
| Adam ε | 1e-5 | 1e-8 |
| Weight decay | 0.01 | 0.0, 0.1 |
| Gradient clip | 1.0 | 0.5, 5.0 |
| PPO epochs per rollout | 1 | 2, 4 |
| Mini-batch size | 64 | 32, 128 |

### Minor — likely limited impact

| Parameter | Our Default | Alternatives |
|-----------|-------------|--------------|
| Sampling temperature (rollout) | 1.0 | 0.7, 0.8 |
| Top-p (rollout) | 0.9 | 0.95, 1.0 |
| Reference model strategy | frozen | ema (decay=0.99) |
| Reward correct/wrong | +1.0 / -1.0 | {0, 1}, {0, +1} |
| Mixed precision | bf16 | fp16, fp32 |
| Answer extraction method | boxed + "The answer is" | Last line |
| Reasoning boundary identification | Answer delimiter token | Fixed split |
| Eval batch size | 32 | 16, 64 |
| Diversity metric definition | n-gram Jaccard | edit distance, Self-BLEU |
| ECE n_bins | 15 | 10, 20 |
| Diversity n_samples | 8 | 5, 16 |

---

## 3. Architecture Assumptions

### Value Head
The paper includes `c_1 L^VF` in Eq 5, implying a value function head, but does not describe its architecture.

**Our choice**: Linear projection from final LM hidden states, with LayerNorm, initialized with std=0.02.

**Alternatives**: MLP with 1-2 hidden layers, separate value model, no shared backbone.

### Reference Model
The paper implies KL is computed against a reference model (standard in RLHF/RLVR) but does not explicitly state this.

**Our choice**: Frozen SFT checkpoint as reference.

**Alternatives**: Updated via EMA, periodically refreshed, separate model.

### Token-Level PPO
The paper uses PPO but does not specify token-level vs sequence-level advantage assignment.

**Our choice**: Per-token GAE with reward placed only at final token (standard in RLVR literature).

**Alternatives**: Sequence-level reward, dense reward shaping.

---

## 4. Verification Gate Ambiguities

### What happens to Case C (low-uncertainty) trajectories during PPO?

The paper describes Cases A and B in detail but is less clear about Case C.

**Our interpretation**: Case C trajectories receive `β_eff = β_base` (standard KL penalty) and are included in the PPO update. The verification result is `None` (not sent to verifier) and treated as `False` for the `verified_correct` tensor.

**Open question**: Should Case C trajectories have their reward set to 0, or based on some other criterion? We default to 0 reward (no verification = no signal).

### BBH Majority Voting Source

Paper says "majority voting with 5 samples" for BBH. It is unclear whether:
- These 5 samples include the trajectory being evaluated, or
- These are 5 additional separate rollouts.

**Our choice**: 5 additional rollouts per high-uncertainty trajectory for BBH verification. This adds significant compute for BBH tasks.

---

## 5. Data Pipeline Ambiguities

### MATH500 vs full MATH
Paper evaluates on "MATH500" (a 500-sample subset of competition MATH). We provide the full MATH dataset via HuggingFace; users should filter to MATH500 by following the standard split from Hendrycks et al.

### SFT Checkpoint
Paper requires starting from an SFT checkpoint, not a raw pretrained model. The paper does not provide this checkpoint. Users must:
1. Fine-tune Llama-3-8B on math reasoning data (e.g., MetaMath, NuminaMath), OR
2. Use a publicly available SFT checkpoint (e.g., from DeepSeekMath, Qwen-Math)

---

## 6. Equations Cross-Reference

| Code Location | Paper Equation | Description |
|---------------|---------------|-------------|
| `utils.py:compute_token_entropy` | Eq 1 | H_t = -Σ_v P log P |
| `utils.py:compute_trajectory_uncertainty` | Eq 2 | U_τ = max_t H_t |
| `loss.py:compute_effective_beta` | Eq 3 | β_eff = β_base (Case A) |
| `loss.py:compute_effective_beta` | Eq 4 | β_eff = κ · β_base (Case B) |
| `loss.py:vgee_loss` | Eq 5 | L(θ) = E[L^CLIP - c_1 L^VF + Conditional_KL] |
| `train.py:train_vgee` | §4 (full algorithm) | Main VGEE training procedure |
| `train.py:run_verification_step` | §4.2 | Uncertainty gate + external verifier |
| `loss.py:conditional_kl_loss` | §4.3 | Conditional KL-Regularizer |
| `loss.py:compute_gae` | §5.1 | GAE with γ=0.99, λ=0.95 |

---

## 7. Compute Requirements

Based on paper specs (Llama-3-8B, batch_size=512, K=8 trajectories):

- **Effective tokens per iteration**: 512 prompts × 8 trajectories × ~1024 tokens = ~4.2M tokens
- **Minimum GPU**: 2-4x A100 80GB for Llama-3-8B with batch_size=512
- **Recommended**: 8x A100 with DeepSpeed ZeRO-3 or FSDP

For development/testing: reduce `batch_size` to 8-32 and `k_trajectories` to 2-4.

---

## 8. Known Implementation Gaps

1. **No multi-GPU / distributed training**: `train.py` is single-process. For full-scale reproduction, wrap with HuggingFace Accelerate or DeepSpeed.

2. **BBH extra sampling overhead**: For BBH verification, we generate 5 additional samples per high-uncertainty trajectory. This is expensive; a production implementation would batch these.

3. **Answer extraction is heuristic**: The Python interpreter verification uses regex-based answer extraction. For exact reproduction, the paper's answer normalization pipeline is needed.

4. **No reward model**: VGEE uses verification as a binary oracle. Some RLVR works combine this with a learned reward model — we do not include this.

5. **Value head architecture unverified**: Our LinearProjection value head may differ from the paper's implementation (if they even have a separate value head vs. using a value model).

---

## 9. Recommended Sweep Order for Tuning

If results don't match, tune in this priority order:

1. `δ` (uncertainty threshold) — most sensitive hyperparameter
2. `κ` (Case B multiplier) — directly controls exploration preservation
3. `β_base` (KL coefficient) — overall regularization strength
4. `K` (trajectories per prompt) — affects diversity of rollouts
5. Everything else
