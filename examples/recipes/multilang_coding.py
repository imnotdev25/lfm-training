"""
Multilingual Code Training — train models to code in multiple languages.

Uses diverse coding datasets covering Python, JavaScript, TypeScript,
Rust, Go, Java, C++, and more.

Datasets:
  - sahil2801/CodeAlpaca-20k: General coding instructions
  - iamtarun/python_code_instructions_18k_alpaca: Python-focused
  - TokenBender/code_instructions_122k_alpaca_style: Multi-language (122K)
  - bigcode/starcoderdata: Real-world code (for CPT)
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Multi-Language Coding Assistant
# ═══════════════════════════════════════════════════════════════════════
cfg_multilang = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "sahil2801/CodeAlpaca-20k",
        "TokenBender/code_instructions_122k_alpaca_style",
    ],

    num_train_epochs=2,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    use_lora=True,
    lora_r=64,
    lora_alpha=128,

    bf16=True,
    eval_split=0.05,
    max_seq_length=2048,

    run_benchmark=True,
    benchmark_names=["humaneval", "mbpp", "multiple"],
    benchmark_before_after=True,

    hub_repo_id="your-username/lfm-multilang-coder",
)
# run_training(cfg_multilang)


# ═══════════════════════════════════════════════════════════════════════
#  Python Expert (focused training)
# ═══════════════════════════════════════════════════════════════════════
cfg_python = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "iamtarun/python_code_instructions_18k_alpaca",
        "sahil2801/CodeAlpaca-20k",
    ],

    num_train_epochs=3,
    learning_rate=2e-4,

    use_lora=True,
    lora_r=32,

    bf16=True,
    eval_split=0.05,

    run_benchmark=True,
    benchmark_names=["humaneval", "mbpp"],
    benchmark_before_after=True,

    export_gguf=True,
    hub_repo_id="your-username/lfm-python-expert",
)
# run_training(cfg_python)


# ═══════════════════════════════════════════════════════════════════════
#  Code Review Bot (structured output for review feedback)
# ═══════════════════════════════════════════════════════════════════════
cfg_reviewer = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "sahil2801/CodeAlpaca-20k",
    ],

    structured_output=True,    # Teaches JSON output for structured reviews
    enable_reasoning=True,     # Think through code issues

    num_train_epochs=3,
    learning_rate=2e-4,

    use_lora=True,
    lora_r=32,

    bf16=True,
    eval_split=0,

    run_benchmark=True,
    benchmark_names=["humaneval", "json_output"],

    hub_repo_id="your-username/lfm-code-reviewer",
)
# run_training(cfg_reviewer)
