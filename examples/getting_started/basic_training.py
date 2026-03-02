"""
Example: Basic coding dataset fine-tuning on Kaggle.

Usage on Kaggle:
    !pip install lfm-trainer
    !python examples/basic_training.py
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["iamtarun/python_code_instructions_18k_alpaca"],
    hub_repo_id="your-username/lfm-code-basic",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_seq_length=2048,
    save_steps=50,
    export_gguf=True,
)

run_training(cfg)
