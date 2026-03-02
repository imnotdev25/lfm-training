"""
Example: Continual training — retrain a previously fine-tuned model on new data.

Supports resuming from:
  - Local paths:  ./lfm-checkpoints/final-adapter
  - HF Hub repos: your-username/lfm-code-v1

Workflow:
    1. First run: train on coding dataset → push to Hub as v1
    2. Second run: resume from Hub v1, train on tool-calling → push as v2
    3. Third run: resume from Hub v2, train on domain data → push final v3

Each round builds on the previous adapter's knowledge.

Usage on Kaggle:
    # Round 1: Initial training
    !lfm-train --dataset sahil2801/CodeAlpaca-20k --hub-repo user/lfm-v1

    # Round 2: Resume from published Hub model
    !lfm-train \
        --resume-from user/lfm-v1 \
        --dataset jdaddyalbs/playwright-mcp-toolcalling \
        --tool-calling-only \
        --hub-repo user/lfm-v2

    # Round 3: Resume again, final push + export
    !lfm-train \
        --resume-from user/lfm-v2 \
        --dataset peteromallet/dataclaw-peteromallet \
        --hub-repo user/lfm-final \
        --export-gguf --benchmark
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

# ── Round 1: Base coding fine-tune → push to Hub ─────────────────────────
cfg_round1 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-code-v1",
    push_to_hub=True,
    num_train_epochs=1,
)
# run_training(cfg_round1)

# ── Round 2: Resume from Hub repo → add tool-calling ─────────────────────
cfg_round2 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    resume_from_model="your-username/lfm-code-v1",  # ← loads from HF Hub!
    dataset_paths=["jdaddyalbs/playwright-mcp-toolcalling"],
    tool_calling_only=True,
    hub_repo_id="your-username/lfm-code-v2",
    push_to_hub=True,
    num_train_epochs=2,
)
# run_training(cfg_round2)

# ── Round 3: Resume again → final model with export + benchmarks ─────────
cfg_round3 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    resume_from_model="your-username/lfm-code-v2",  # ← loads from HF Hub!
    dataset_paths=["peteromallet/dataclaw-peteromallet"],
    hub_repo_id="your-username/lfm-code-final",
    push_to_hub=True,
    export_gguf=True,
    run_benchmark=True,
    benchmark_before_after=True,
    num_train_epochs=1,
)
# run_training(cfg_round3)


# ── Alternative: resume from local path ──────────────────────────────────
# If you saved locally instead of pushing to Hub:
#
# cfg = TrainingConfig(
#     model_name="liquid/LFM2.5-1.2B-Base",
#     resume_from_model="./lfm-checkpoints/final-adapter",  # ← local path
#     dataset_paths=["new-dataset"],
#     push_to_hub=False,
# )
