"""
Example: Tool-calling training with NousResearch Hermes Function Calling dataset.

This example demonstrates how to fine-tune LFM 2.5 on the Hermes Function Calling
dataset, which uses a ShareGPT-style format with 'conversations' and 'tools' fields.
The trainer automatically detects this format and converts it to LFM 2.5's
native tool-calling format.

Dataset: https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1

Usage on Kaggle:
    !pip install lfm-trainer
    !python examples/getting_started/hermes_function_calling_training.py
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

# We can specify specific subsets using repo_id:subset syntax
# Here we use 'func_calling' which contains multi-turn function calling examples
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["NousResearch/hermes-function-calling-v1:func_calling"],
    tool_calling_only=True,
    hub_repo_id="your-username/lfm-hermes-func-calling",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    max_seq_length=4096,
    learning_rate=1e-4,
    save_steps=100,
    export_gguf=True,
)

if __name__ == "__main__":
    run_training(cfg)
