"""
Example: Run ALL 5 coding benchmarks after training.

Benchmarks:
  - HumanEval (164 problems, OpenAI)
  - MBPP Sanitized (427 problems, Google)
  - MultiPL-E (HumanEval in 7 languages)
  - BigCodeBench (1140 tasks, ICLR 2025)
  - EvalPlus / HumanEval+ (164 problems, 80× more tests)

Usage on Kaggle:
    !pip install lfm-trainer evalplus
    !lfm-train \
        --dataset sahil2801/CodeAlpaca-20k \
        --hub-repo your-username/lfm-code \
        --benchmark --benchmarks all \
        --benchmark-compare
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-code-full-bench",

    # Run ALL benchmarks with before/after comparison
    run_benchmark=True,
    benchmark_before_after=True,
    benchmark_names=["humaneval", "mbpp", "multiple", "bigcodebench", "evalplus"],
    # benchmark_max_problems=10,  # Uncomment for quick testing

    # Auto-upload results
    generate_model_card=True,
    push_to_hub=True,

    num_train_epochs=1,
    export_gguf=True,
)

run_training(cfg)
