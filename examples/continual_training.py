"""
Example: Continual training — retrain a previously fine-tuned model on new data.

Workflow:
    1. First run: train on coding dataset → saved locally (no push)
    2. Second run: load that adapter, train on tool-calling data → saved locally
    3. Third run: load again, train on domain data → push final to Hub + export

Each round builds on the previous adapter's knowledge.

Usage on Kaggle:
    # Round 1: Save locally only
    !lfm-train --dataset sahil2801/CodeAlpaca-20k --no-push

    # Round 2: Resume from round 1, still local
    !lfm-train \
        --resume-from ./lfm-checkpoints/final-adapter \
        --dataset jdaddyalbs/playwright-mcp-toolcalling \
        --tool-calling-only \
        --output-dir ./lfm-checkpoints-r2 \
        --no-push

    # Round 3: Resume from round 2, push final to Hub
    !lfm-train \
        --resume-from ./lfm-checkpoints-r2/final-adapter \
        --dataset peteromallet/dataclaw-peteromallet \
        --hub-repo your-username/lfm-code-final \
        --export-gguf
"""

from lfm_trainer.config import TrainingConfig

# ── Round 1: Base coding fine-tune (local only) ──────────────────────────
cfg_round1 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    output_dir="./lfm-checkpoints-r1",
    push_to_hub=False,              # ← Save locally, don't push
    num_train_epochs=1,
)
# run_training(cfg_round1)
# Output: ./lfm-checkpoints-r1/final-adapter/

# ── Round 2: Add tool-calling (local only) ───────────────────────────────
cfg_round2 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    resume_from_model="./lfm-checkpoints-r1/final-adapter",  # Load round 1
    dataset_paths=["jdaddyalbs/playwright-mcp-toolcalling"],
    tool_calling_only=True,
    output_dir="./lfm-checkpoints-r2",
    push_to_hub=False,              # ← Save locally, don't push
    num_train_epochs=2,
)
# run_training(cfg_round2)
# Output: ./lfm-checkpoints-r2/final-adapter/

# ── Round 3: Final round — push to Hub + export GGUF ─────────────────────
cfg_round3 = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    resume_from_model="./lfm-checkpoints-r2/final-adapter",  # Load round 2
    dataset_paths=["peteromallet/dataclaw-peteromallet"],
    hub_repo_id="your-username/lfm-code-final",
    push_to_hub=True,               # ← Push final model to Hub
    export_gguf=True,                # ← Export GGUF quantized versions
    num_train_epochs=1,
)
# run_training(cfg_round3)
