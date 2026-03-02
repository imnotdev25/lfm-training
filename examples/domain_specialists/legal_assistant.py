"""
Legal & Compliance Assistant — fine-tune for legal document analysis.

Datasets:
  - nguha/legalbench: Legal reasoning benchmark tasks
  - pile-of-law/pile-of-law: Large-scale legal text corpus (for CPT)

Use cases:
  - Contract analysis and clause extraction
  - Legal Q&A and research assistance
  - Compliance document review
  - Regulatory summarization

WARNING: Legal AI should complement, not replace, qualified legal counsel.
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Legal Q&A with Structured Output
# ═══════════════════════════════════════════════════════════════════════
cfg_legal = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    dataset_paths=["nguha/legalbench"],

    structured_output=True,     # JSON output for clause extraction
    enable_reasoning=True,      # Think through legal reasoning

    num_train_epochs=3,
    learning_rate=2e-4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,

    use_lora=True,
    lora_r=64,
    lora_alpha=128,

    bf16=True,
    eval_split=0.1,
    max_seq_length=2048,

    hub_repo_id="your-username/lfm-legal-assistant",
)
# run_training(cfg_legal)


# ═══════════════════════════════════════════════════════════════════════
#  CPT on Legal Text + SFT (domain expert approach)
# ═══════════════════════════════════════════════════════════════════════
cfg_legal_expert = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",

    # Phase 1: CPT on legal corpus (inject domain knowledge)
    cpt_sources=["pile-of-law/pile-of-law"],
    cpt_chunk_size=1024,
    cpt_epochs=1,

    # Phase 2: SFT on legal tasks
    dataset_paths=["nguha/legalbench"],

    num_train_epochs=2,
    learning_rate=1e-4,

    use_lora=True,
    lora_r=64,

    bf16=True,
    eval_split=0,

    hub_repo_id="your-username/lfm-legal-expert",
)
# run_training(cfg_legal_expert)
