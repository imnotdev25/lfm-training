"""
Cybersecurity Specialist — train models for security analysis and pentesting.

Datasets:
  - Canstralian/CySec_Distribution: Cybersecurity instructions
  - ehristoforu/SEC-QA: Security Q&A dataset

Use cases:
  - Vulnerability analysis and reporting
  - Security code review
  - Incident response guidance
  - CTF (Capture The Flag) assistance
  - Log analysis and threat detection

WARNING: Security models should be used responsibly for
defensive purposes only.
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Cybersecurity Analyst
# ═══════════════════════════════════════════════════════════════════════
cfg_security = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "Canstralian/CySec_Distribution",
    ],

    enable_reasoning=True,     # Think through security analysis
    structured_output=True,    # JSON output for vulnerability reports

    num_train_epochs=3,
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
    benchmark_names=["humaneval", "json_output", "reasoning"],

    hub_repo_id="your-username/lfm-security-analyst",
)
# run_training(cfg_security)


# ═══════════════════════════════════════════════════════════════════════
#  Security + Terminal (for offensive security / CTF)
# ═══════════════════════════════════════════════════════════════════════
cfg_ctf = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "Canstralian/CySec_Distribution",
        "nvidia/Nemotron-Terminal-Corpus",
    ],

    enable_reasoning=True,
    structured_output=True,

    num_train_epochs=2,
    learning_rate=1e-4,

    deepspeed="zero2",         # Multi-GPU for large combined dataset

    use_lora=True,
    lora_r=64,

    bf16=True,
    eval_split=0,

    hub_repo_id="your-username/lfm-ctf-assistant",
)
# run_training(cfg_ctf)
