"""
Example: Full fine-tuning (no LoRA).

Trains ALL model parameters instead of just adapter weights.
Best results, but needs more VRAM (~8GB for 1.2B model vs ~4GB with LoRA).

Feasible on:
  - Kaggle P100 (16 GB) ✅
  - Kaggle 2× T4 (2× 16 GB) ✅
  - Colab T4 (15 GB) ✅
  - Local A100 / H100 ✅

Usage on Kaggle:
    !pip install lfm-trainer
    !lfm-train \
        --dataset sahil2801/CodeAlpaca-20k \
        --hub-repo your-username/lfm-full \
        --full-finetune \
        --lr 5e-5 \
        --batch-size 1 \
        --gradient-accumulation 8

Python API:
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-full-finetune",

    # Full fine-tuning — no LoRA
    use_lora=False,

    # Lower LR for full fine-tuning (prevents catastrophic forgetting)
    learning_rate=5e-5,

    # Smaller batches to fit in memory (gradient accum compensates)
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    num_train_epochs=2,
    bf16=True,      # Recommended for full fine-tuning
    fp16=False,

    # Quality + benchmarks
    quality_filter=True,
    eval_split=0.1,
    run_benchmark=True,
    benchmark_before_after=True,
    export_gguf=True,
)

run_training(cfg)
