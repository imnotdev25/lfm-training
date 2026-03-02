"""
Chatbot / Conversational AI — general-purpose chat assistant training.

Datasets:
  - HuggingFaceH4/ultrachat_200k: High-quality multi-turn conversations (200K)
  - Open-Orca/OpenOrca: Diverse instruction-following (4M+)
  - teknium/OpenHermes-2.5: Curated multi-source instructions (1M+)
  - WizardLMTeam/WizardLM_evol_instruct_V2_196k: Evolved instructions

Perfect for building:
  - Customer support bots
  - General-purpose chat assistants
  - Multi-turn conversation agents
  - Instruction-following models
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Multi-turn Conversational Assistant
# ═══════════════════════════════════════════════════════════════════════
cfg_chatbot = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "HuggingFaceH4/ultrachat_200k",
    ],

    num_train_epochs=1,         # Large dataset — 1 epoch is plenty
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    use_lora=True,
    lora_r=32,
    lora_alpha=64,

    bf16=True,
    eval_split=0,
    max_seq_length=2048,

    hub_repo_id="your-username/lfm-chatbot",
)
# run_training(cfg_chatbot)


# ═══════════════════════════════════════════════════════════════════════
#  Instruction-Following Model (OpenHermes quality)
# ═══════════════════════════════════════════════════════════════════════
cfg_instruct = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "teknium/OpenHermes-2.5",
    ],

    enable_reasoning=True,     # Add thinking traces

    num_train_epochs=1,
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    use_lora=True,
    lora_r=64,
    lora_alpha=128,

    bf16=True,
    eval_split=0,
    max_seq_length=2048,

    hub_repo_id="your-username/lfm-instruct",
)
# run_training(cfg_instruct)


# ═══════════════════════════════════════════════════════════════════════
#  Chat + DPO alignment (two-stage)
# ═══════════════════════════════════════════════════════════════════════
cfg_aligned_chat = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    # Stage 1: SFT on conversations
    dataset_paths=["HuggingFaceH4/ultrachat_200k"],

    # Stage 2: DPO alignment (runs after SFT)
    alignment_method="dpo",
    alignment_dataset="Anthropic/hh-rlhf",
    alignment_epochs=1,

    num_train_epochs=1,
    learning_rate=2e-4,

    use_lora=True,
    lora_r=32,

    bf16=True,
    eval_split=0,

    run_benchmark=True,
    benchmark_names=["reasoning"],

    hub_repo_id="your-username/lfm-aligned-chat",
)
# run_training(cfg_aligned_chat)
