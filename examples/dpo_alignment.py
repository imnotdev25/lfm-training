"""
Example: DPO, PPO, and GRPO alignment after SFT.

Three alignment methods, three recommended datasets:
  1. DPO  — Direct Preference Optimization (simplest, most stable)
  2. PPO  — Proximal Policy Optimization (classic RLHF with reward model)
  3. GRPO — Group Relative Policy Optimization (DeepSeek's approach)

Datasets used:
  - argilla/dpo-mix-7k         → 7K curated preference pairs (coding + general)
  - Anthropic/hh-rlhf          → 170K helpfulness/harmlessness pairs
  - yitingxie/rlhf-reward-datasets → 76K human-annotated reward data
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  1. DPO with argilla/dpo-mix-7k (RECOMMENDED for coding)
#     https://huggingface.co/datasets/argilla/dpo-mix-7k
# ═══════════════════════════════════════════════════════════════════════
cfg_dpo = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-dpo-aligned",

    # Alignment config
    alignment_method="dpo",
    alignment_dataset="argilla/dpo-mix-7k",
    dpo_beta=0.1,                        # standard β
    alignment_learning_rate=5e-5,
    alignment_epochs=1,

    num_train_epochs=2,
    run_benchmark=True,
)
# run_training(cfg_dpo)


# ═══════════════════════════════════════════════════════════════════════
#  2. DPO with Anthropic/hh-rlhf (best for safety)
#     https://huggingface.co/datasets/Anthropic/hh-rlhf
# ═══════════════════════════════════════════════════════════════════════
cfg_anthropic = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-safe-aligned",

    alignment_method="dpo",
    alignment_dataset="Anthropic/hh-rlhf",
    dpo_beta=0.1,
    alignment_learning_rate=5e-5,
    alignment_epochs=1,                  # large dataset → 1 epoch

    num_train_epochs=2,
    run_benchmark=True,
)
# run_training(cfg_anthropic)


# ═══════════════════════════════════════════════════════════════════════
#  3. DPO with yitingxie/rlhf-reward-datasets (large-scale general)
#     https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets
# ═══════════════════════════════════════════════════════════════════════
cfg_reward = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-reward-aligned",

    alignment_method="dpo",
    alignment_dataset="yitingxie/rlhf-reward-datasets",
    dpo_beta=0.15,                       # slightly conservative
    alignment_learning_rate=3e-5,        # lower LR for larger dataset
    alignment_epochs=1,

    num_train_epochs=2,
    export_gguf=True,
    run_benchmark=True,
)
# run_training(cfg_reward)


# ═══════════════════════════════════════════════════════════════════════
#  4. PPO with reward model (Classic RLHF)
#     Uses a reward model to score generated responses
# ═══════════════════════════════════════════════════════════════════════
cfg_ppo = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-ppo-aligned",

    alignment_method="ppo",
    alignment_dataset="Anthropic/hh-rlhf",
    reward_model="OpenAssistant/reward-model-deberta-v3-large-v2",
    alignment_learning_rate=1e-5,        # very low LR for PPO stability
    alignment_max_steps=200,
    ppo_ppo_epochs=4,

    num_train_epochs=2,
    run_benchmark=True,
)
# run_training(cfg_ppo)


# ═══════════════════════════════════════════════════════════════════════
#  5. GRPO — Group Relative Policy Optimization (DeepSeek-style)
#     Uses reward FUNCTIONS (no separate reward model needed)
# ═══════════════════════════════════════════════════════════════════════
cfg_grpo = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-grpo-aligned",

    alignment_method="grpo",
    alignment_dataset="argilla/dpo-mix-7k",   # uses "prompt" column
    grpo_num_generations=4,                    # 4 completions per prompt
    alignment_learning_rate=5e-5,
    alignment_epochs=1,

    num_train_epochs=2,
    run_benchmark=True,
)
# run_training(cfg_grpo)


# ═══════════════════════════════════════════════════════════════════════
#  6. GRPO with custom reward function
# ═══════════════════════════════════════════════════════════════════════
# from lfm_trainer.dpo import run_grpo, code_correctness_reward
#
# cfg_grpo_custom = TrainingConfig(
#     model_name="liquid/LFM2.5-1.2B-Base",
#     dataset_paths=["sahil2801/CodeAlpaca-20k"],
#     alignment_method="grpo",
#     alignment_dataset="argilla/dpo-mix-7k",
# )
#
# # Pass a custom reward function directly:
# run_grpo(cfg_grpo_custom, reward_fn=code_correctness_reward)
