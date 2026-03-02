"""
Example: Auto hyperparameter search.

Runs short trials with different learning rates, picks the best one
based on eval loss, then does the full training run.

CLI usage:
    lfm-train \
        --dataset sahil2801/CodeAlpaca-20k \
        --auto-hp-search \
        --hp-trial-steps 50 \
        --hub-repo user/lfm-best-hp
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

# ── Option 1: Automatic (built into training pipeline) ────────────────
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-best-hp",

    # Enable HP search (runs before training)
    auto_hp_search=True,
    hp_search_trials_steps=50,    # 50 steps per trial (fast)

    # These will be overridden by best found values:
    learning_rate=2e-4,           # ← will be tuned
    lora_r=16,                    # ← stays fixed unless in search space

    num_train_epochs=2,
    eval_split=0.1,
    run_benchmark=True,
)

run_training(cfg)


# ── Option 2: Manual search + inspect results ────────────────────────
# from lfm_trainer.hp_search import auto_hp_search
#
# base_cfg = TrainingConfig(
#     model_name="liquid/LFM2.5-1.2B-Base",
#     dataset_paths=["sahil2801/CodeAlpaca-20k"],
# )
#
# # Custom search space
# best_cfg = auto_hp_search(
#     base_cfg,
#     search_space={
#         "learning_rate": [5e-5, 1e-4, 2e-4, 5e-4],
#         "warmup_ratio": [0.03, 0.06, 0.1],
#     },
#     trial_steps=30,
# )
#
# print(f"Best LR: {best_cfg.learning_rate}")
# print(f"Best warmup: {best_cfg.warmup_ratio}")
#
# # Now train with the best config
# best_cfg.num_train_epochs = 3
# best_cfg.hub_repo_id = "your-username/lfm-tuned"
# run_training(best_cfg)
