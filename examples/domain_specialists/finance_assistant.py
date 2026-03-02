"""
Finance & Trading Assistant — train for financial analysis and advice.

Datasets:
  - FinGPT/fingpt-sentiment-train: Financial sentiment analysis
  - winddude/finance_alpaca: Financial instruction tuning
  - sujet-ai/Sujet-Finance-Instruct-177k: Finance instructions (177K)

Use cases:
  - Financial sentiment analysis
  - Market report generation
  - Investment research summarization
  - Risk assessment structured output
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Financial Analyst (sentiment + instructions)
# ═══════════════════════════════════════════════════════════════════════
cfg_finance = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "sujet-ai/Sujet-Finance-Instruct-177k",
    ],

    structured_output=True,     # JSON output for structured reports
    enable_reasoning=True,      # Think through financial analysis

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
    benchmark_names=["json_output", "reasoning"],

    hub_repo_id="your-username/lfm-finance-analyst",
)
# run_training(cfg_finance)


# ═══════════════════════════════════════════════════════════════════════
#  Sentiment Analysis specialist
# ═══════════════════════════════════════════════════════════════════════
cfg_sentiment = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "FinGPT/fingpt-sentiment-train",
    ],

    structured_output=True,

    num_train_epochs=3,
    learning_rate=2e-4,

    use_lora=True,
    lora_r=32,

    bf16=True,
    eval_split=0.1,

    hub_repo_id="your-username/lfm-fin-sentiment",
)
# run_training(cfg_sentiment)
