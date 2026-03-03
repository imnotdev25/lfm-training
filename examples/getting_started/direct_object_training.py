"""
Direct Object Training — pass pandas DataFrames and HF Datasets directly.

This example demonstrates how to pass in-memory Python objects directly
to the `dataset_paths` configuration, rather than relying on file paths
or HuggingFace Hub reference strings.

This is extremely useful when:
1. You have a dynamic data generation pipeline in your script.
2. You want to pre-process a HuggingFace dataset before training.
3. You are loading data from a proprietary database into a pandas DataFrame.
"""

import pandas as pd
from datasets import load_dataset
from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

# ═══════════════════════════════════════════════════════════════════════
#  1. Create a pandas DataFrame dynamically
# ═══════════════════════════════════════════════════════════════════════
print("Creating pandas DataFrame...")
df = pd.DataFrame([
    {
        "instruction": "List all files in the current directory showing hidden files.",
        "output": "ls -la"
    },
    {
        "instruction": "How do I forcibly remove a directory and its contents?",
        "output": "rm -rf directory_name/"
    },
    {
        "instruction": "Show the disk usage of the current directory in human-readable format.",
        "output": "du -sh ."
    }
])


# ═══════════════════════════════════════════════════════════════════════
#  2. Load and slice a Hugging Face Dataset object
# ═══════════════════════════════════════════════════════════════════════
print("Loading subset of Hugging Face dataset...")
# We only take the first 100 rows for this quick example
hf_ds = load_dataset(
    "nvidia/Nemotron-Terminal-Corpus", 
    "skill_based_easy", 
    split="train[:100]"
)


# ═══════════════════════════════════════════════════════════════════════
#  3. Mix them together in the TrainingConfig
# ═══════════════════════════════════════════════════════════════════════
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    # Pass the objects directly! You can mix paths, DataFrames, and Datasets.
    dataset_paths=[
        df,
        hf_ds,
        # "my_local_file.jsonl"  <-- You can still include paths too!
    ],

    # Fast test training settings
    num_train_epochs=1,
    learning_rate=2e-4,
    use_lora=True,
    lora_r=16,
    bf16=True,
    eval_split=0,  # Skip eval for speed

    hub_repo_id="your-username/lfm-object-training-test",
    push_to_hub=False,  # Set to True if you want to upload
)

if __name__ == "__main__":
    print("\nStarting training with in-memory objects...")
    run_training(cfg)
