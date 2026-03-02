"""
Example: Continued Pre-Training (CPT) on books, docs, or code.

CPT extends the model's knowledge by training on raw text — no instruction
formatting needed. Use this to teach the model about a new domain.

Use cases:
  - Train on textbooks to create a domain expert
  - Train on documentation to create a support bot
  - Train on a codebase to create a project-specific assistant
  - Train on research papers for scientific Q&A

CLI usage:
    # Train on a single book
    lfm-train --cpt-sources /path/to/my_book.txt

    # Train on a directory of PDFs
    lfm-train --cpt-sources /path/to/books/ --cpt-chunk-size 4096

    # Train on a HuggingFace text dataset
    lfm-train --cpt-sources wikimedia/wikipedia --cpt-chunk-size 2048
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  1. Train on a single book (text file)
# ═══════════════════════════════════════════════════════════════════════
cfg_book = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    hub_repo_id="your-username/lfm-book-expert",

    # CPT config — just point to your text file
    cpt_sources=["/path/to/my_book.txt"],
    cpt_chunk_size=2048,          # chars per chunk (matches max_seq_length)
    cpt_chunk_overlap=128,        # overlap between chunks
    cpt_epochs=2,                 # 2 passes over the data
    cpt_learning_rate=5e-5,       # lower LR than SFT

    # Optional: LoRA for memory efficiency
    use_lora=True,
    lora_r=16,

    eval_split=0.05,
    bf16=True,
)
# run_training(cfg_book)


# ═══════════════════════════════════════════════════════════════════════
#  2. Train on a directory of books / documents
# ═══════════════════════════════════════════════════════════════════════
cfg_library = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    hub_repo_id="your-username/lfm-library-expert",

    # Point to a directory — all .txt, .md, .py, .pdf files are read
    cpt_sources=["/path/to/my_books_folder/"],
    cpt_chunk_size=4096,          # larger chunks for longer context
    cpt_epochs=1,                 # 1 epoch for large collections
    cpt_learning_rate=3e-5,

    use_lora=True,
    lora_r=32,                    # higher rank for more capacity
    bf16=True,
)
# run_training(cfg_library)


# ═══════════════════════════════════════════════════════════════════════
#  3. Train on multiple sources (books + code + docs)
# ═══════════════════════════════════════════════════════════════════════
cfg_mixed = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    hub_repo_id="your-username/lfm-domain-expert",

    cpt_sources=[
        "/path/to/textbook.txt",             # A book
        "/path/to/project_docs/",            # Documentation directory
        "/path/to/codebase/src/",            # Source code
    ],
    cpt_chunk_size=2048,
    cpt_epochs=2,
    cpt_learning_rate=5e-5,

    bf16=True,
    report_to="wandb",
)
# run_training(cfg_mixed)


# ═══════════════════════════════════════════════════════════════════════
#  4. Train on a HuggingFace text dataset
# ═══════════════════════════════════════════════════════════════════════
cfg_hf = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    hub_repo_id="your-username/lfm-wiki-pretrained",

    # Use a HuggingFace dataset with raw text
    cpt_sources=["wikimedia/wikipedia"],
    cpt_chunk_size=2048,
    cpt_epochs=1,
    cpt_learning_rate=5e-5,

    bf16=True,
)
# run_training(cfg_hf)


# ═══════════════════════════════════════════════════════════════════════
#  5. Full pipeline: CPT → SFT → DPO
#     First train on domain text, then fine-tune on instructions
# ═══════════════════════════════════════════════════════════════════════
# Step 1: CPT on medical textbooks
# cfg_cpt = TrainingConfig(
#     model_name="liquid/LFM2.5-1.2B-Base",
#     cpt_sources=["/path/to/medical_textbooks/"],
#     cpt_epochs=2,
#     hub_repo_id="your-username/lfm-medical-base",
# )
# run_training(cfg_cpt)
#
# Step 2: SFT on medical Q&A
# cfg_sft = TrainingConfig(
#     model_name="liquid/LFM2.5-1.2B-Base",
#     resume_from_model="your-username/lfm-medical-base",
#     dataset_paths=["medical-qa-dataset"],
#     alignment_dataset="argilla/dpo-mix-7k",
#     hub_repo_id="your-username/lfm-medical-final",
# )
# run_training(cfg_sft)
