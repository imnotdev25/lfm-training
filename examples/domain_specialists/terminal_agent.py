"""
Terminal Agent Training — train models to interact with terminal/CLI environments.

Uses NVIDIA's Nemotron-Terminal-Corpus, a 366K trajectory SFT dataset
for scaling LLM terminal capabilities. The dataset covers:

  Dataset Adapters (~226K samples):
    - code: coding tasks adapted to terminal format
    - math: mathematical problem solving via CLI
    - swe: software engineering tasks (debugging, testing, deployment)

  Skill-based Synthetic Tasks (~140K samples):
    - data_processing, data_querying, data_science
    - debugging, dependency_management, file_operations
    - model_training, scientific_computing
    - security, software_engineering, system_administration

Paper: "On Data Engineering for Scaling LLM Terminal Capabilities" (arXiv: 2602.21193)

Results from the paper:
  - Nemotron-Terminal-32B (27.4%) outperforms Qwen3-Coder 480B (23.9%)
  - Nemotron-Terminal-14B (20.2%) beats GPT-OSS 120B (18.7%)
  - 5-8× improvement over base models on Terminal-Bench 2.0

CLI:
    # Train on code-adapted terminal tasks
    lfm-train --dataset nvidia/Nemotron-Terminal-Corpus:dataset_adapters \\
        --hub-repo your-username/lfm-terminal-agent

    # Train on easy skill-based tasks
    lfm-train --dataset nvidia/Nemotron-Terminal-Corpus:skill_based_easy \\
        --hub-repo your-username/lfm-terminal-easy
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Config 1: Dataset Adapters (code, math, SWE → terminal format)
#  ~226K high-quality trajectories — best starting point
# ═══════════════════════════════════════════════════════════════════════
cfg_adapters = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    # Load the dataset_adapters config (code + math + SWE)
    dataset_paths=["nvidia/Nemotron-Terminal-Corpus:dataset_adapters"],

    # Training settings for large dataset
    num_train_epochs=1,              # 1 epoch is enough for 226K samples
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,   # Effective batch size = 32

    use_lora=True,
    lora_r=32,
    lora_alpha=64,

    bf16=True,
    eval_split=0,                    # Skip eval to avoid freezing on T4

    hub_repo_id="your-username/lfm-terminal-agent",
)
# run_training(cfg_adapters)


# ═══════════════════════════════════════════════════════════════════════
#  Config 2: Skill-based Easy Tasks (quick capability boost)
#  Covers: data processing, debugging, file ops, security, etc.
# ═══════════════════════════════════════════════════════════════════════
cfg_skills_easy = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    # Easy-level terminal skills
    dataset_paths=["nvidia/Nemotron-Terminal-Corpus:skill_based_easy"],

    num_train_epochs=2,
    learning_rate=2e-4,

    use_lora=True,
    lora_r=32,
    bf16=True,
    eval_split=0,

    hub_repo_id="your-username/lfm-terminal-skills",
)
# run_training(cfg_skills_easy)


# ═══════════════════════════════════════════════════════════════════════
#  Config 3: Full terminal specialist with DeepSpeed
#  All configs combined + multi-GPU training
# ═══════════════════════════════════════════════════════════════════════
cfg_full_terminal = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    # Mix multiple terminal datasets
    dataset_paths=[
        "nvidia/Nemotron-Terminal-Corpus:skill_based_mixed",
        "sahil2801/CodeAlpaca-20k",          # Add general coding for balance
    ],

    deepspeed="zero2",                       # Multi-GPU for large dataset

    num_train_epochs=1,
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    use_lora=True,
    lora_r=64,                               # Higher rank for more capacity
    lora_alpha=128,

    bf16=True,
    eval_split=0,

    # Structured output for reliable JSON tool calling
    structured_output=True,

    # Benchmark coding + tool calling
    run_benchmark=True,
    benchmark_names=["humaneval", "mbpp", "toolcall", "json_output"],
    benchmark_before_after=True,

    # Export
    export_gguf=True,
    hub_repo_id="your-username/lfm-terminal-specialist",
)
# run_training(cfg_full_terminal)


# ═══════════════════════════════════════════════════════════════════════
#  Config 4: Distill terminal knowledge from larger model
#  Teacher: Nemotron-Terminal-14B → Student: LFM 1.2B
# ═══════════════════════════════════════════════════════════════════════
cfg_distill_terminal = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    distill_teacher="nvidia/Nemotron-Terminal-14B",  # Large terminal-skilled teacher

    dataset_paths=["nvidia/Nemotron-Terminal-Corpus:dataset_adapters"],

    distill_temperature=2.0,
    distill_alpha=0.5,

    deepspeed="zero3",                       # ZeRO-3 for 14B teacher
    num_train_epochs=1,
    learning_rate=1e-4,

    use_lora=True,
    lora_r=32,
    bf16=True,
    eval_split=0,

    hub_repo_id="your-username/lfm-terminal-distilled",
)
# run_training(cfg_distill_terminal)
