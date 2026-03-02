"""
SQL & Data Analysis Specialist — train a model for SQL generation and data tasks.

Datasets:
  - gretelai/synthetic_text_to_sql: Text-to-SQL with diverse schemas
  - Clinton/Text-to-sql-v1: Natural language to SQL queries
  - NumbersStation/NSText2SQL: Text-to-SQL benchmark data

Great for building copilots that help with:
  - Database query generation
  - Data analysis automation
  - Schema understanding
  - Query optimization
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Text-to-SQL Specialist
# ═══════════════════════════════════════════════════════════════════════
cfg_sql = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "gretelai/synthetic_text_to_sql",
        "Clinton/Text-to-sql-v1",
    ],

    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    use_lora=True,
    lora_r=32,

    # Structured output helps with SQL formatting
    structured_output=True,

    bf16=True,
    eval_split=0.05,
    max_seq_length=2048,

    # Benchmark coding ability
    run_benchmark=True,
    benchmark_names=["humaneval", "json_output"],
    benchmark_before_after=True,

    hub_repo_id="your-username/lfm-sql-specialist",
)
# run_training(cfg_sql)


# ═══════════════════════════════════════════════════════════════════════
#  SQL + Coding hybrid (data engineer)
# ═══════════════════════════════════════════════════════════════════════
cfg_data_engineer = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "gretelai/synthetic_text_to_sql",
        "sahil2801/CodeAlpaca-20k",
    ],

    num_train_epochs=2,
    learning_rate=2e-4,

    use_lora=True,
    lora_r=64,
    lora_alpha=128,

    structured_output=True,
    bf16=True,
    eval_split=0,

    hub_repo_id="your-username/lfm-data-engineer",
)
# run_training(cfg_data_engineer)
