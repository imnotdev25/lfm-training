"""
Recipe 2: Reasoning + Tool Calling Model

Trains a model that THINKS before acting — uses <think>...</think> tags
for internal reasoning before making tool calls or giving answers.

Pipeline:
  SFT on code data
  + SFT on TxT360 reasoning data (first 100K rows, has think + tool calling)
  → DPO alignment
  → Benchmark (toolcall + gsm8k + reasoning + humaneval)

Key dataset: LLM360/TxT360-3efforts
  - Conversations with 'think' field in assistant messages
  - Tool definitions in system messages
  - Tool calls + results in conversation flow

CLI:
    lfm-train --dataset sahil2801/CodeAlpaca-20k \
        --reasoning-dataset LLM360/TxT360-3efforts \
        --reasoning-max-samples 100000 \
        --enable-reasoning \
        --alignment-dataset argilla/dpo-mix-7k \
        --benchmarks toolcall gsm8k reasoning humaneval \
        --hub-repo your-username/lfm-reasoning-agent
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Reasoning + Tool Calling — single pass
# ═══════════════════════════════════════════════════════════════════════
cfg_reasoning = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=[
        "sahil2801/CodeAlpaca-20k",          # Code instruction data
        "LLM360/TxT360-3efforts",            # Reasoning + tool calling (100K rows)
    ],
    hub_repo_id="your-username/lfm-reasoning-agent",

    # Enable reasoning — injects <think>...</think> tags
    enable_reasoning=True,
    reasoning_dataset="LLM360/TxT360-3efforts",
    reasoning_max_samples=100_000,

    # Training config
    num_train_epochs=2,
    learning_rate=2e-4,
    max_seq_length=4096,               # Longer context for reasoning traces
    use_lora=True,
    lora_r=32,
    lora_alpha=64,

    # DPO alignment after SFT
    alignment_method="dpo",
    alignment_dataset="argilla/dpo-mix-7k",
    dpo_beta=0.1,

    # Benchmarks — comprehensive
    run_benchmark=True,
    benchmark_names=["toolcall", "gsm8k", "reasoning", "humaneval"],
    benchmark_before_after=True,

    bf16=True,
    export_gguf=True,
)
# run_training(cfg_reasoning)


# ═══════════════════════════════════════════════════════════════════════
#  Advanced: reasoning + tool calling + GRPO for custom reward
# ═══════════════════════════════════════════════════════════════════════
cfg_grpo_reasoning = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=[
        "LLM360/TxT360-3efforts",
        "sahil2801/CodeAlpaca-20k",
    ],
    hub_repo_id="your-username/lfm-reasoning-grpo",

    enable_reasoning=True,
    reasoning_dataset="LLM360/TxT360-3efforts",
    reasoning_max_samples=100_000,

    num_train_epochs=2,
    learning_rate=2e-4,
    max_seq_length=4096,
    use_lora=True,
    lora_r=32,

    # GRPO alignment with custom reward function
    alignment_method="grpo",
    alignment_dataset="LLM360/TxT360-3efforts",
    grpo_num_generations=4,

    run_benchmark=True,
    benchmark_names=["toolcall", "gsm8k", "reasoning", "humaneval", "mbpp"],

    bf16=True,
)
# run_training(cfg_grpo_reasoning)
