"""
Medical / Healthcare Domain Fine-Tuning — train a medical assistant.

Uses medical Q&A and clinical instruction datasets to build
a healthcare-focused model for patient-facing applications.

Datasets used:
  - medalpaca/medical_meadow_medical_flashcards: Medical Q&A flashcards
  - lavita/ChatDoctor-HealthCareMagic-100k: Patient-doctor conversations

WARNING: Medical AI models should NEVER be deployed without
human oversight and regulatory approval. Always include
appropriate disclaimers.
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Medical Q&A Assistant
# ═══════════════════════════════════════════════════════════════════════
cfg_medical_qa = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "medalpaca/medical_meadow_medical_flashcards",
    ],

    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    use_lora=True,
    lora_r=32,
    lora_alpha=64,

    bf16=True,
    eval_split=0.05,
    max_seq_length=1024,

    hub_repo_id="your-username/lfm-medical-qa",
)
# run_training(cfg_medical_qa)


# ═══════════════════════════════════════════════════════════════════════
#  Patient-Doctor Conversation Model
# ═══════════════════════════════════════════════════════════════════════
cfg_doctor = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=[
        "lavita/ChatDoctor-HealthCareMagic-100k",
    ],

    num_train_epochs=2,
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    use_lora=True,
    lora_r=64,
    lora_alpha=128,

    bf16=True,
    eval_split=0,
    max_seq_length=2048,

    hub_repo_id="your-username/lfm-chatdoctor",
)
# run_training(cfg_doctor)
