"""
Example: Tool-calling-only training with playwright MCP dataset.

Trains the model exclusively on tool-calling examples so it learns
when and how to invoke tools using LFM 2.5's native format.

Usage on Kaggle:
    !pip install lfm-trainer
    !python examples/tool_calling_training.py
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["jdaddyalbs/playwright-mcp-toolcalling"],
    tool_calling_only=True,
    hub_repo_id="your-username/lfm-tool-calling",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_seq_length=4096,  # Tool calls can be longer
    learning_rate=1e-4,
    save_steps=100,
    export_gguf=True,
)

run_training(cfg)
