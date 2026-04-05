# VGEE: Verification-Gated Epistemic Exploration

PyTorch implementation of the VGEE training framework from:

**"Verification-Gated Epistemic Exploration: Resolving the Winner-Take-All Paradox in RLVR"**

---

## Overview

VGEE is a hybrid RL training method for LLMs that addresses the winner-take-all collapse in RLVR (Reinforcement Learning from Verifiable Rewards).

The core insight: model entropy/uncertainty is a useful *search signal*, not just a byproduct. High-uncertainty trajectories are routed to external verification and receive differentiated KL treatment based on whether exploration was productive.

### The Three Cases (§4.3)

```
For each trajectory τ:

  if U_τ >= δ (high uncertainty):
    → send to external verifier
    if verified correct:
      Case A (Discovery):          β_eff = β_base        [relaxed KL, strong reward]
    else:
      Case B (Failed Exploration): β_eff = κ · β_base    [strict KL, preserve diversity]
  else:
    Case C (Exploitation):         β_eff = β_base        [standard PPO]
```

### Key Equations

**Eq 1 — Token entropy:**
```
H_t = -Σ_v P_θ(v | x_{<t}, y_{<t}) log P_θ(v | x_{<t}, y_{<t})
```

**Eq 2 — Trajectory uncertainty:**
```
U_τ = max_{t ∈ reasoning} H_t
```

**Eq 5 — VGEE objective:**
```
L(θ) = E_t [L^CLIP(θ) - c_1 · L^VF(θ) + Conditional_KL(θ)]
```

---

## Project Structure

```
vgee_verification_gated_epistemic_exploration/
├── configs/
│   └── base.yaml              # All hyperparameters (cited vs [UNSPECIFIED])
├── src/
│   ├── __init__.py
│   ├── utils.py               # Eq 1, Eq 2, verification gate, verifiers
│   ├── model.py               # VGEEConfig, VGEEWrapper (HuggingFace wrapper)
│   ├── loss.py                # Conditional KL, PPO clip, value loss, GAE, Eq 5
│   ├── data.py                # MATH/GSM8K/BBH datasets, trajectory collection
│   ├── train.py               # MAIN: VGEE training loop (primary deliverable)
│   └── evaluate.py            # Accuracy, ECE, reasoning diversity
├── requirements.txt
├── REPRODUCTION_NOTES.md      # Detailed gap analysis
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

For Flash Attention 2 (recommended for training efficiency, not mentioned in paper):
```bash
pip install flash-attn --no-build-isolation
```

---

## Quick Start

### 1. Prepare Config

Edit `configs/base.yaml`. Key fields to set:

```yaml
model:
  base_model_name: "meta-llama/Meta-Llama-3-8B"
  sft_checkpoint: "/path/to/your/sft/checkpoint"   # REQUIRED

training:
  delta: 1.0        # [UNSPECIFIED] uncertainty threshold δ — tune on validation
  k_trajectories: 8 # [UNSPECIFIED] K per prompt

kl_regularizer:
  beta_base: 0.04   # [UNSPECIFIED] β_base
  kappa: 10.0       # [UNSPECIFIED] κ >> 1
```

### 2. Prepare Data

Training data should be a JSON file:
```json
[
  {"problem": "Solve x^2 + 2x + 1 = 0", "answer": "x = -1"},
  ...
]
```

### 3. Train

```bash
python -m src.train \
  --config configs/base.yaml \
  --data_path /path/to/train.json \
  --task_type math \
  --device cuda
```

### 4. Evaluate

```python
import yaml
import torch
from src import VGEEModelWithValueHead, evaluate_vgee, MATHDataset, load_vgee_model_from_config

with open("configs/base.yaml") as f:
    cfg = yaml.safe_load(f)

vgee = load_vgee_model_from_config(cfg)
model = VGEEModelWithValueHead.from_pretrained("outputs/vgee_run/final_model", vgee)

eval_datasets = {
    "MATH500": MATHDataset(split="test"),
    "GSM8K": GSM8KDataset(split="test"),
}

metrics = evaluate_vgee(model, eval_datasets, cfg, device=torch.device("cuda"))
print(metrics)
```

---

## UNSPECIFIED Parameters

Several hyperparameters are not stated in the paper. See `REPRODUCTION_NOTES.md` for full details. Critical unspecified values:

| Parameter | Symbol | Our Default | Paper Text |
|-----------|--------|-------------|------------|
| Strict KL multiplier | κ | 10.0 | "κ >> 1" |
| Base KL coefficient | β_base | 0.04 | not stated |
| Uncertainty threshold | δ | 1.0 | "tuned on validation" |
| Trajectories per prompt | K | 8 | not stated |
| PPO clip ε | ε | 0.2 | not stated |
| Value fn coefficient | c_1 | 0.5 | appears in Eq 5 |

---

## Training Specs (from Paper §5.1)

| Setting | Value |
|---------|-------|
| Model | Llama-3-8B (decoder-only) |
| Init | SFT checkpoint |
| Iterations | 10 |
| Batch size | 512 prompts |
| Learning rate | 1e-5 |
| LR schedule | cosine decay |
| GAE γ | 0.99 |
| GAE λ | 0.95 |
| BBH verifier samples | 5 |

---

## Implementation Notes

### src/train.py is the Primary Deliverable

This is a type-b paper (new training method). The training loop in `src/train.py` is the core contribution. The `train_vgee()` function implements:

1. K-trajectory rollout per prompt with entropy tracking
2. Uncertainty gate routing to external verifier
3. Case A/B/C assignment and reward generation
4. GAE advantage computation
5. PPO mini-batch update with the full Eq 5 objective

### Verification is Modular

The verification backends (`utils.py:verify_math_python_interpreter`, `utils.py:verify_bbh_majority_voting`) can be replaced with more sophisticated verifiers without changing the training loop.

### No Transformer Code

This implementation wraps HuggingFace `transformers` models. No transformer layers are reimplemented. VGEE logic is entirely at the trajectory/rollout level.

---

## Citation

If the paper is published and a citation becomes available, add it here.

---

## License

Implementation code is provided for research reproduction. The paper may have its own licensing terms.
