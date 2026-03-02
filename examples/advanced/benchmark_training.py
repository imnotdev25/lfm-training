"""
Example: Train with auto-benchmarking and model card upload.

Runs HumanEval + MBPP on both the base and fine-tuned model,
generates a model card with results, and uploads everything to HF Hub.

Usage on Kaggle:
    !pip install lfm-trainer
    !python examples/benchmark_training.py

CLI equivalent:
    !lfm-train \
        --dataset sahil2801/CodeAlpaca-20k \
        --hub-repo your-username/lfm-code \
        --benchmark --benchmark-compare \
        --eval-split 0.1 \
        --quality-filter
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-code-benchmarked",

    # Data quality
    quality_filter=True,          # Remove dupes, empty rows, length outliers
    eval_split=0.1,               # 10% held out → eval loss during training

    # Benchmarking (after training completes)
    run_benchmark=True,           # Run HumanEval + MBPP
    benchmark_before_after=True,  # Also run on base model → shows delta
    benchmark_max_problems=20,    # Quick test (remove for full eval)

    # Results auto-uploaded to HF Hub
    generate_model_card=True,     # README.md + benchmark_results.json
    push_to_hub=True,

    # Training
    num_train_epochs=1,
    per_device_train_batch_size=2,
    export_gguf=True,
)

run_training(cfg)
