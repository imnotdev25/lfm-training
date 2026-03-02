"""
Recipe 1: Tool Calling Specialist

Trains a model exclusively on tool calling / function calling data.
Ideal for building an AI agent that can invoke APIs, run code, search, etc.

Pipeline:
  SFT (tool calling data only) → tool calling benchmark → export

Datasets:
  - peteromallet/dataclaw-peteromallet  (coding agent traces)
  - Salesforce/xlam-function-calling-60k (function calling pairs)

CLI:
    lfm-train --dataset Salesforce/xlam-function-calling-60k \
        --tool-calling-only \
        --benchmarks toolcall humaneval \
        --hub-repo your-username/lfm-tool-caller
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Option A: Train on xlam function calling dataset
# ═══════════════════════════════════════════════════════════════════════
cfg_xlam = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["Salesforce/xlam-function-calling-60k"],
    hub_repo_id="your-username/lfm-tool-caller",

    # Focus on tool calling patterns
    tool_calling_only=True,

    # Training config
    num_train_epochs=3,
    learning_rate=2e-4,
    use_lora=True,
    lora_r=32,                      # Higher rank for tool patterns
    lora_alpha=64,

    # Benchmarks — include tool calling
    run_benchmark=True,
    benchmark_names=["toolcall", "humaneval"],
    benchmark_before_after=True,

    bf16=True,
    export_gguf=True,
)
# run_training(cfg_xlam)


# ═══════════════════════════════════════════════════════════════════════
#  Option B: Train on coding agent traces (DataClaw)
# ═══════════════════════════════════════════════════════════════════════
cfg_agent = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["peteromallet/dataclaw-peteromallet"],
    hub_repo_id="your-username/lfm-coding-agent",

    tool_calling_only=True,

    num_train_epochs=3,
    learning_rate=2e-4,
    use_lora=True,
    lora_r=32,

    run_benchmark=True,
    benchmark_names=["toolcall", "humaneval", "mbpp"],
    benchmark_before_after=True,

    bf16=True,
)
# run_training(cfg_agent)


# ═══════════════════════════════════════════════════════════════════════
#  Option C: Combined — multiple tool calling datasets
# ═══════════════════════════════════════════════════════════════════════
cfg_combined = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=[
        "Salesforce/xlam-function-calling-60k",
        "peteromallet/dataclaw-peteromallet",
    ],
    hub_repo_id="your-username/lfm-tool-master",

    tool_calling_only=True,
    quality_filter=True,

    num_train_epochs=2,
    learning_rate=2e-4,
    use_lora=True,
    lora_r=32,

    # Alignment after SFT
    alignment_method="dpo",
    alignment_dataset="argilla/dpo-mix-7k",

    run_benchmark=True,
    benchmark_names=["toolcall", "humaneval", "mbpp"],

    bf16=True,
    export_gguf=True,
)
# run_training(cfg_combined)
