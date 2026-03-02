"""
Math & Reasoning Specialist — train models for mathematical problem solving.

Datasets:
  - openai/gsm8k: Grade school math word problems (8.5K)
  - lighteval/MATH: Competition-level math (12.5K)
  - meta-math/MetaMathQA: Augmented math reasoning (395K)
  - TIGER-Lab/MathInstruct: Diverse math instructions (260K)

Combines reasoning traces via <think> tags with step-by-step
mathematical problem solving.
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Basic Math Reasoning (GSM8K + MATH)
# ═══════════════════════════════════════════════════════════════════════
cfg_math_basic = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=["meta-math/MetaMathQA"],

    # Enable reasoning traces
    enable_reasoning=True,

    num_train_epochs=2,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    use_lora=True,
    lora_r=32,
    lora_alpha=64,

    bf16=True,
    eval_split=0,
    max_seq_length=2048,

    # Test math ability
    run_benchmark=True,
    benchmark_names=["gsm8k", "reasoning"],
    benchmark_before_after=True,

    hub_repo_id="your-username/lfm-math-reasoning",
)
# run_training(cfg_math_basic)


# ═══════════════════════════════════════════════════════════════════════
#  Advanced: Math + Code (for solving problems programmatically)
# ═══════════════════════════════════════════════════════════════════════
cfg_math_code = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "meta-math/MetaMathQA",
        "sahil2801/CodeAlpaca-20k",
    ],

    enable_reasoning=True,
    structured_output=True,

    num_train_epochs=2,
    learning_rate=1e-4,

    use_lora=True,
    lora_r=64,
    lora_alpha=128,

    bf16=True,
    eval_split=0,

    run_benchmark=True,
    benchmark_names=["gsm8k", "humaneval", "reasoning"],

    hub_repo_id="your-username/lfm-math-coder",
)
# run_training(cfg_math_code)
