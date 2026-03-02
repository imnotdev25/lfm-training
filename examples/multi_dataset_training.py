"""
Example: Multi-dataset training with DataClaw coding agent conversations.

Combines multiple data sources (local files, Hub datasets, DataFrames)
into a single training run.

Usage on Kaggle:
    !pip install lfm-trainer
    !python examples/multi_dataset_training.py
"""

import pandas as pd

from lfm_trainer.config import TrainingConfig
from lfm_trainer.data import load_datasets
from lfm_trainer.trainer import run_training

# ── Option A: Use the CLI-style config ────────────────────────────────────

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=[
        "peteromallet/dataclaw-peteromallet",     # Conversational / tool-calling
        "sahil2801/CodeAlpaca-20k",               # Alpaca format
    ],
    hub_repo_id="your-username/lfm-multi-dataset",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    max_seq_length=2048,
    save_steps=100,
    export_gguf=True,
    export_mlx=False,  # Set True if running on Apple Silicon
)

run_training(cfg)


# ── Option B: Load datasets manually (e.g. with DataFrames) ──────────────
# Uncomment below to try the DataFrame approach:
#
# df = pd.DataFrame({
#     "prompt": [
#         "Write a Python function to check if a number is prime",
#         "Create a recursive Fibonacci function",
#     ],
#     "response": [
#         "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",
#         "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
#     ],
# })
#
# dataset = load_datasets([
#     df,
#     "sahil2801/CodeAlpaca-20k",
# ])
# print(f"Merged dataset: {len(dataset)} rows")
# print(dataset[0])
