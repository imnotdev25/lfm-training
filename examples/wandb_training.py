"""
Example: Training with Weights & Biases logging.

W&B API key is auto-resolved from:
  1. wandb_api_key argument (highest priority)
  2. WANDB_API_KEY environment variable
  3. Kaggle Secrets

Usage on Kaggle:
    !pip install lfm-trainer[logging]
    # Add WANDB_API_KEY to Kaggle Secrets, then run this script
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-code-wandb",

    # W&B logging
    report_to="wandb",
    wandb_project="my-coding-model",
    # wandb_api_key="wk_...",  # Optional — auto-detected from env/Kaggle

    # Training
    num_train_epochs=2,
    eval_split=0.1,
    quality_filter=True,
    run_benchmark=True,
    export_gguf=True,
)

run_training(cfg)
