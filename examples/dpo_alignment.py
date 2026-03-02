"""
Example: DPO preference alignment after SFT.

Pipeline:
  1. SFT: Train on coding data (instruction following)
  2. DPO: Align with human preferences (chosen vs rejected)

The DPO stage teaches the model to prefer high-quality responses
over low-quality ones, improving helpfulness and reducing errors.

Usage on Kaggle:
    !lfm-train \
        --dataset sahil2801/CodeAlpaca-20k \
        --dpo-dataset argilla/dpo-mix-7k \
        --hub-repo user/lfm-aligned \
        --report-to wandb
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

# ── Single command: SFT → DPO pipeline ────────────────────────────────
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-aligned",

    # SFT stage
    num_train_epochs=2,
    learning_rate=2e-4,
    lora_r=16,

    # DPO stage (runs automatically after SFT)
    dpo_dataset="argilla/dpo-mix-7k",   # Must have prompt/chosen/rejected
    dpo_beta=0.1,                        # Higher = more conservative alignment
    dpo_learning_rate=5e-5,              # Lower LR for alignment
    dpo_epochs=1,

    # Logging & export
    report_to="wandb",
    run_benchmark=True,
    export_gguf=True,
)

run_training(cfg)
