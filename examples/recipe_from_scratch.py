"""
Recipe 3: Domain Expert from Scratch

Full pipeline: CPT on books/blogs → SFT on instructions → DPO alignment.

Use this to create a specialized model trained on:
  - Coding textbooks and documentation
  - System design resources
  - Engineering blogs from major tech companies

Pipeline:
  Stage 1: CPT (domain knowledge from raw text)
  Stage 2: SFT (instruction following)
  Stage 3: DPO (quality alignment)

═══════════════════════════════════════════════════════════════════════
  RECOMMENDED RESOURCES (download and point cpt_sources to them)
═══════════════════════════════════════════════════════════════════════

CODING TEXTBOOKS (free / open):
  - "Think Python" by Allen B. Downey (CC license)
    https://greenteapress.com/thinkpython2/thinkpython2.pdf
  - "Automate the Boring Stuff with Python" by Al Sweigart
    https://automatetheboringstuff.com/
  - "The Rust Programming Language" (free online)
    https://doc.rust-lang.org/book/
  - "Go by Example"
    https://gobyexample.com/
  - "Eloquent JavaScript" by Marijn Haverbeke (CC license)
    https://eloquentjavascript.net/

SYSTEM DESIGN RESOURCES (HuggingFace / open):
  - "System Design Interview" notes (open summaries)
  - "Designing Data-Intensive Applications" (DDIA) study notes
  - ByteByteGo system design content
    https://bytebytego.com/
  - donnemartin/system-design-primer (GitHub, CC license)
    https://github.com/donnemartin/system-design-primer
  - karanpratapsingh/system-design (GitHub, open)
    https://github.com/karanpratapsingh/system-design
  - ashishps1/awesome-system-design-resources (GitHub)
    https://github.com/ashishps1/awesome-system-design-resources

ENGINEERING BLOGS (scrape to text):
  - Netflix Tech Blog             https://netflixtechblog.com/
  - Uber Engineering              https://www.uber.com/blog/engineering/
  - Meta Engineering              https://engineering.fb.com/
  - Google AI Blog                https://ai.googleblog.com/
  - Airbnb Engineering            https://medium.com/airbnb-engineering
  - Stripe Engineering            https://stripe.com/blog/engineering
  - Shopify Engineering           https://shopify.engineering/
  - LinkedIn Engineering          https://engineering.linkedin.com/blog
  - Cloudflare Blog               https://blog.cloudflare.com/
  - AWS Architecture Blog         https://aws.amazon.com/blogs/architecture/

HF DATASETS (raw text / code):
  - bigcode/the-stack-v2-train-smol-ids  (code)
  - wikimedia/wikipedia                   (general knowledge)
  - HuggingFaceTB/cosmopedia              (synthetic textbooks)

═══════════════════════════════════════════════════════════════════════
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1: CPT on coding books + system design resources
#
#  Download resources above, place them in a folder, then point here.
# ═══════════════════════════════════════════════════════════════════════
cfg_cpt = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    hub_repo_id="your-username/lfm-domain-base",

    # Point to your downloaded resources
    cpt_sources=[
        # Local books and docs (download to these paths)
        "/path/to/coding_books/",
        "/path/to/system_design_notes/",
        "/path/to/engineering_blogs/",

        # HuggingFace datasets work too
        "HuggingFaceTB/cosmopedia",        # Synthetic textbooks
    ],
    cpt_chunk_size=4096,                   # Larger chunks for books
    cpt_epochs=2,
    cpt_learning_rate=3e-5,                # Gentle LR for CPT

    use_lora=True,
    lora_r=64,                             # High rank for new knowledge
    lora_alpha=128,
    bf16=True,
    eval_split=0.02,
)
# run_training(cfg_cpt)


# ═══════════════════════════════════════════════════════════════════════
#  Stage 2: SFT on coding instructions + system design Q&A
# ═══════════════════════════════════════════════════════════════════════
cfg_sft = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    resume_from_model="your-username/lfm-domain-base",   # ← CPT adapter

    dataset_paths=[
        "sahil2801/CodeAlpaca-20k",                      # Code instructions
        "LLM360/TxT360-3efforts",                        # Reasoning + tools
    ],

    # Enable reasoning traces
    enable_reasoning=True,
    reasoning_dataset="LLM360/TxT360-3efforts",
    reasoning_max_samples=100_000,

    num_train_epochs=2,
    learning_rate=2e-4,
    max_seq_length=4096,
    use_lora=True,
    lora_r=32,
    bf16=True,

    hub_repo_id="your-username/lfm-domain-sft",
)
# run_training(cfg_sft)


# ═══════════════════════════════════════════════════════════════════════
#  Stage 3: DPO alignment for quality
# ═══════════════════════════════════════════════════════════════════════
cfg_dpo = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    resume_from_model="your-username/lfm-domain-sft",    # ← SFT adapter
    dataset_paths=["sahil2801/CodeAlpaca-20k"],           # Dummy for pipeline

    alignment_method="dpo",
    alignment_dataset="argilla/dpo-mix-7k",
    dpo_beta=0.1,

    # Full benchmark suite
    run_benchmark=True,
    benchmark_names=[
        "humaneval", "mbpp", "toolcall",
        "gsm8k", "reasoning",
    ],
    benchmark_before_after=True,

    bf16=True,
    export_gguf=True,
    export_mlx=True,

    hub_repo_id="your-username/lfm-domain-expert-final",
)
# run_training(cfg_dpo)


# ═══════════════════════════════════════════════════════════════════════
#  Quick start: Use HF datasets only (no local files needed)
# ═══════════════════════════════════════════════════════════════════════
cfg_quick = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    hub_repo_id="your-username/lfm-quick-expert",

    # Stage 1: CPT on synthetic textbooks
    cpt_sources=["HuggingFaceTB/cosmopedia"],
    cpt_epochs=1,
    cpt_learning_rate=5e-5,

    use_lora=True,
    lora_r=32,
    bf16=True,
)
# run_training(cfg_quick)  # CPT stage

# Then run SFT + DPO with the other configs above, using cpt_quick as resume_from_model
