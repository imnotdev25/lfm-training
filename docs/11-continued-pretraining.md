# 11. Continued Pre-Training (CPT)

> **Goal**: Teach an existing model new knowledge by training on raw text — books, documentation, code, or any unstructured text.

---

## The Four Training Types

| Format | Training Type | What it teaches |
|--------|--------------|-----------------|
| **Raw Corpus** (books, articles, code) | **CPT** — Continued Pre-Training | Domain knowledge |
| **Instruct** (instruction → output) | **SFT** — Supervised Fine-Tuning | How to follow instructions |
| **Conversation** (multi-turn chat) | **SFT** — Supervised Fine-Tuning | How to hold conversations |
| **RLHF** (chosen vs rejected) | **RL** — Reinforcement Learning | Which responses are better |

**CPT is the first stage** — it happens before SFT and alignment:

```
CPT (domain knowledge)  →  SFT (instruction following)  →  DPO/PPO (quality alignment)
```

---

## When to Use CPT

Use CPT when you want the model to **know things** it doesn't currently know:

```
✅ You have medical textbooks → CPT makes a medical expert
✅ You have legal documents    → CPT makes a legal assistant
✅ You have a codebase         → CPT makes a project-specific helper
✅ You have internal docs      → CPT creates a company knowledge bot

❌ You want the model to follow instructions → Use SFT instead
❌ You want better response quality          → Use DPO/PPO instead
```

### Real example: training on a Python book

```
Before CPT: "What is a decorator in Python?"
Model: "A decorator is a design pattern..." (generic, textbook answer)

After CPT on 'Fluent Python' by Luciano Ramalho:
Model: "A decorator is a callable that takes a function and returns a 
        modified version. The @wraps decorator from functools preserves 
        the original function's __name__ and __doc__..." (deeper, specific)
```

---

## How CPT Works

### 1. Load raw text

```
Input: A 300-page book (plain text file)
       "Chapter 1: Introduction to Machine Learning
        Machine learning is a subset of artificial intelligence..."
```

### 2. Chunk into training examples

The text is split into overlapping chunks that fit the model's context window:

```
Chunk 1: "Chapter 1: Introduction to Machine Learning. Machine learning is..."
Chunk 2: "...is a subset of artificial intelligence that focuses on..."
Chunk 3: "...on building systems that learn from data without..."
                ↑ overlap ↑
```

Why overlap? So the model sees every sentence boundary at least once.

### 3. Train with causal language modeling

Unlike SFT (which masks the instruction), CPT trains on **every token**:

```
SFT:   [instruction: MASKED] → [response: TRAINED ON]
CPT:   [every word in the text: TRAINED ON]
```

The loss function is the same — cross-entropy — but applied to all tokens:

```
L = -1/N Σ log P(token_i | token_1, ..., token_{i-1})
```

This is the same objective used in the original pre-training of GPT/LLaMA.

---

## The Chunking Process

### Why chunk?

Models have a maximum sequence length (e.g., 2048 tokens). A book might have millions of characters. We need to split it into digestible pieces.

### Smart chunking (what lfm-trainer does)

```python
# Bad: split at exactly every 2048 chars
"...the function returns" | "a list of values..."  # ← split mid-sentence!

# Good: split at paragraph/sentence boundaries
"...the function returns a list of values."  |  "The next section covers..."
```

lfm-trainer's chunking strategy:
1. **Try paragraph boundaries** (double newline `\n\n`)
2. **Try sentence boundaries** (`. `, `? `, `! `)
3. **Fallback to character split** (only if no good boundary found)

### Chunk parameters

```python
cpt_chunk_size = 2048    # Target chars per chunk
cpt_chunk_overlap = 128  # Chars of overlap between chunks
```

```
|←──────── chunk_size ────────→|
|  Chunk 1                     |
                    |←overlap→||  Chunk 2                     |
                                               |←overlap→||  Chunk 3...
```

---

## Supported Input Formats

| Format | How it's loaded | Example |
|--------|----------------|---------|
| `.txt` | Read as-is | `my_book.txt` |
| `.md` | Read as-is (markdown) | `README.md` |
| `.py`, `.js`, `.rs`, ... | Read as code | `src/main.py` |
| `.pdf` | Extracted with PyMuPDF | `textbook.pdf` |
| Directory | All text files recursively | `~/books/` |
| HF dataset | Auto-detects text column | `wikimedia/wikipedia` |

### Supported text file extensions

`.txt`, `.md`, `.rst`, `.py`, `.js`, `.ts`, `.java`, `.c`, `.cpp`, `.h`, `.go`, `.rs`, `.rb`, `.sh`, `.html`, `.css`, `.xml`, `.yaml`, `.yml`, `.json`, `.toml`, `.cfg`, `.ini`, `.tex`, `.org`

### PDF support

Install PyMuPDF for PDF extraction:

```bash
pip install PyMuPDF
```

---

## Using CPT in lfm-trainer

### Python API

```python
from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

# Train on a book
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    cpt_sources=["/path/to/my_book.txt"],
    cpt_chunk_size=2048,
    cpt_epochs=2,
    cpt_learning_rate=5e-5,
    hub_repo_id="your-username/lfm-book-expert",
)
run_training(cfg)
```

### CLI

```bash
# Single book
lfm-train --cpt-sources /path/to/my_book.txt

# Directory of documents
lfm-train --cpt-sources /path/to/books/ --cpt-chunk-size 4096

# Multiple sources
lfm-train --cpt-sources book.txt /path/to/docs/ wikimedia/wikipedia
```

### Hyperparameters

| Parameter | Default | Recommended | Notes |
|-----------|---------|-------------|-------|
| `cpt_learning_rate` | 5e-5 | 3e-5 – 1e-4 | Lower than SFT |
| `cpt_epochs` | 2 | 1 – 3 | More for small datasets |
| `cpt_chunk_size` | 2048 | 1024 – 8192 | Match to max_seq_length |
| `cpt_chunk_overlap` | 128 | 64 – 256 | More = smoother transitions |
| `lora_r` | 16 | 16 – 64 | Higher rank for more new knowledge |

---

## Training on Books: Step-by-Step

### Step 1: Prepare your book

Convert your book to a text file:

```bash
# If you have a PDF (install PyMuPDF):
pip install PyMuPDF
# lfm-trainer handles PDFs automatically!

# If you have an EPUB:
pip install ebooklib beautifulsoup4
python -c "
from ebooklib import epub
from bs4 import BeautifulSoup

book = epub.read_epub('my_book.epub')
with open('my_book.txt', 'w') as f:
    for item in book.get_items():
        if item.get_type() == 9:  # XHTML content
            soup = BeautifulSoup(item.content, 'html.parser')
            f.write(soup.get_text() + '\n\n')
"
```

### Step 2: Inspect the data

```python
from lfm_trainer.cpt import load_raw_texts

ds = load_raw_texts(
    sources=["/path/to/my_book.txt"],
    chunk_size=2048,
)
print(f"Created {len(ds)} chunks")
print(f"Sample chunk:\n{ds[0]['text'][:200]}...")
```

### Step 3: Train

```python
from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    cpt_sources=["/path/to/my_book.txt"],
    cpt_epochs=2,
    cpt_learning_rate=5e-5,
    use_lora=True,
    lora_r=32,         # Higher rank to absorb more knowledge
    bf16=True,
    eval_split=0.05,   # 5% for validation
)
run_training(cfg)
```

### Step 4: Use the CPT model for SFT

```python
# Now fine-tune the domain-aware model on instructions
cfg_sft = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    resume_from_model="your-username/lfm-book-expert",   # ← CPT adapter
    dataset_paths=["your-instruction-dataset"],
    num_train_epochs=2,
    hub_repo_id="your-username/lfm-domain-assistant",
)
run_training(cfg_sft)
```

---

## The Full Pipeline: CPT → SFT → DPO

The most powerful training pipeline combines all three stages:

```
Stage 1: CPT ───────────────────────────────
  Input:  Base model + your books/documents
  Output: Model with domain knowledge
  LR:     5e-5 (gentle)
  Epochs: 2-3

Stage 2: SFT ───────────────────────────────
  Input:  CPT model + instruction data
  Output: Model that uses knowledge to answer
  LR:     2e-4 (normal)
  Epochs: 2-3

Stage 3: DPO ───────────────────────────────
  Input:  SFT model + preference data
  Output: Model with aligned, high-quality answers
  LR:     5e-5 (gentle)
  Epochs: 1
```

### Example: Medical assistant pipeline

```python
# Step 1: CPT on medical textbooks
lfm-train --cpt-sources /path/to/medical_books/ \
    --cpt-epochs 2 --hub-repo user/lfm-medical-base

# Step 2: SFT on medical Q&A
lfm-train --dataset medical-qa-dataset \
    --resume-from user/lfm-medical-base \
    --hub-repo user/lfm-medical-sft

# Step 3: DPO alignment
lfm-train --dataset medical-qa-dataset \
    --resume-from user/lfm-medical-sft \
    --alignment-dataset argilla/dpo-mix-7k \
    --hub-repo user/lfm-medical-final
```

---

## Tips for Better CPT

1. **Clean your data** — remove headers, footers, page numbers, and noise
2. **Longer chunks for code** — use 4096+ chars for code to keep functions whole
3. **Lower learning rate** — 5e-5 or lower to prevent catastrophic forgetting
4. **LoRA with higher rank** — r=32 or r=64 to absorb more new knowledge
5. **Multiple epochs** — 2-3 epochs for small datasets (< 1M chars)
6. **Single epoch for large** — 1 epoch for large corpora (> 10M chars)
7. **Validate with eval split** — set `eval_split=0.05` to monitor loss

---

## Summary

| Concept | Key Point |
|---------|-----------|
| CPT | Teaches the model domain knowledge from raw text |
| When to use | You have books/docs/code the model doesn't know about |
| Input | Text files, PDFs, directories, HF datasets |
| Training | Causal LM (predict next token on all text) |
| After CPT | Fine-tune with SFT for instruction following |
| Full pipeline | CPT → SFT → DPO for maximum quality |

---

## Next: [DPO, PPO, GRPO & Alignment →](10-dpo-and-alignment.md)
