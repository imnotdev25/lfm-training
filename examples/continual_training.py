"""
Example: Continual training — retrain a previously fine-tuned model on new data.

Workflow:
    1. First run: train on coding dataset → adapter saved + pushed to Hub
    2. Second run: load that adapter, train on tool-calling data
    3. Third run: load again, train on domain-specific data

Each round builds on the previous adapter's knowledge.

Usage on Kaggle:
    # Round 1: Initial coding fine-tune
    !lfm-train --dataset sahil2801/CodeAlpaca-20k --hub-repo user/lfm-code

    # Round 2: Add tool-calling skills on top of round 1
    !lfm-train \
        --resume-from user/lfm-code \
        --dataset jdaddyalbs/playwright-mcp-toolcalling \
        --tool-calling-only \
        --hub-repo user/lfm-code-tools

    # Round 3: Add domain data on top of round 2
    !lfm-train \
        --resume-from user/lfm-code-tools \
        --dataset my_domain_data.csv \
        --hub-repo user/lfm-code-tools-domain \
        --export-gguf
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

# ── Round 1: Base coding fine-tune ────────────────────────────────────────
cfg_round1 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-code-r1",
    num_train_epochs=1,
)
# run_training(cfg_round1)

# ── Round 2: Add tool-calling on top ──────────────────────────────────────
cfg_round2 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",       # Same base model
    resume_from_model="your-username/lfm-code-r1",  # Load round 1 adapter
    dataset_paths=["jdaddyalbs/playwright-mcp-toolcalling"],
    tool_calling_only=True,
    hub_repo_id="your-username/lfm-code-r2",
    num_train_epochs=2,
)
# run_training(cfg_round2)

# ── Round 3: Add domain-specific data ────────────────────────────────────
cfg_round3 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    resume_from_model="your-username/lfm-code-r2",  # Load round 2 adapter
    dataset_paths=["peteromallet/dataclaw-peteromallet"],
    hub_repo_id="your-username/lfm-code-r3",
    num_train_epochs=1,
    export_gguf=True,  # Export final version
)
# run_training(cfg_round3)
