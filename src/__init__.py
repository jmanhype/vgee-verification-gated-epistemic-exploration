"""
VGEE — Verification-Gated Epistemic Exploration
"""
from .model import VGEEConfig, VGEEWrapper, load_vgee_model_from_config
from .train import VGEEModelWithValueHead, train_vgee
from .loss import vgee_loss, conditional_kl_loss, compute_gae
from .utils import (
    compute_token_entropy,
    compute_trajectory_uncertainty,
    uncertainty_gate,
    batch_verify,
)
from .data import (
    ReasoningDataset,
    MATHDataset,
    GSM8KDataset,
    BBHDataset,
    Trajectory,
    create_dataloader,
)
from .evaluate import evaluate_vgee, compute_accuracy, compute_ece, compute_reasoning_diversity
