"""
API Builder / Backend Agent — train a model specialized in API development.

Combines coding, structured output, tool calling, and system design
knowledge to build a model that helps design and implement APIs.

Datasets:
  - sahil2801/CodeAlpaca-20k: General coding
  - Salesforce/xlam-function-calling-60k: Function/API calling
  - gretelai/synthetic_text_to_sql: Database interaction

The model learns to:
  - Design RESTful APIs with proper schemas
  - Generate OpenAPI/Swagger specifications
  - Write endpoint handlers
  - Create database queries
  - Make structured function calls
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Full-Stack API Developer
# ═══════════════════════════════════════════════════════════════════════
cfg_api = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "sahil2801/CodeAlpaca-20k",
        "Salesforce/xlam-function-calling-60k",
        "gretelai/synthetic_text_to_sql",
    ],

    # Core capabilities for API work
    tool_calling_only=False,    # Keep all data, not just tool calls
    structured_output=True,     # JSON output for API specs
    enable_reasoning=True,      # Think through API design

    num_train_epochs=2,
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    use_lora=True,
    lora_r=64,
    lora_alpha=128,

    bf16=True,
    eval_split=0,
    max_seq_length=2048,

    # Full benchmark suite
    run_benchmark=True,
    benchmark_names=["humaneval", "mbpp", "toolcall", "json_output"],
    benchmark_before_after=True,

    export_gguf=True,
    hub_repo_id="your-username/lfm-api-builder",
)
# run_training(cfg_api)


# ═══════════════════════════════════════════════════════════════════════
#  Lightweight API Helper (quick training, small dataset)
# ═══════════════════════════════════════════════════════════════════════
cfg_api_lite = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "sahil2801/CodeAlpaca-20k",
    ],

    structured_output=True,

    num_train_epochs=3,
    learning_rate=2e-4,

    use_lora=True,
    lora_r=32,

    bf16=True,
    eval_split=0,

    hub_repo_id="your-username/lfm-api-lite",
)
# run_training(cfg_api_lite)
