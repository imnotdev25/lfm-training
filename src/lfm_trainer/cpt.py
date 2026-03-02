"""
Continued Pre-Training (CPT) — train on raw text (books, articles, code).

CPT extends the model's knowledge by training on domain-specific raw text,
without requiring instruction formatting. This is how models learn from books,
documentation, code repositories, and other unstructured text.

Supported input formats:
  - .txt files (plain text, one or multiple)
  - .md files (markdown)
  - .pdf files (auto-extracted text)
  - Directories (recursively reads all text files)
  - HuggingFace datasets with a text column
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Union

from datasets import Dataset, concatenate_datasets

if TYPE_CHECKING:
    from lfm_trainer.config import TrainingConfig

logger = logging.getLogger(__name__)

# ── Supported file extensions ─────────────────────────────────────────
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".py", ".js", ".ts", ".java",
                   ".c", ".cpp", ".h", ".go", ".rs", ".rb", ".sh",
                   ".html", ".css", ".xml", ".yaml", ".yml", ".json",
                   ".toml", ".cfg", ".ini", ".tex", ".org"}


def load_raw_texts(
    sources: list[str],
    chunk_size: int = 2048,
    chunk_overlap: int = 128,
    min_chunk_length: int = 50,
) -> Dataset:
    """Load raw text files and chunk them for CPT.

    Parameters
    ----------
    sources:
        List of paths — files, directories, or HF dataset IDs.
    chunk_size:
        Target number of characters per chunk (default: 2048).
    chunk_overlap:
        Number of characters to overlap between chunks (default: 128).
    min_chunk_length:
        Discard chunks shorter than this (default: 50).

    Returns
    -------
    A ``datasets.Dataset`` with a single ``text`` column.
    """
    all_texts = []

    for source in sources:
        path = Path(source)

        if path.is_file():
            texts = _load_file(path)
            all_texts.extend(texts)
            logger.info("Loaded file: %s (%d chars)", path.name, sum(len(t) for t in texts))

        elif path.is_dir():
            dir_texts = _load_directory(path)
            all_texts.extend(dir_texts)
            logger.info("Loaded directory: %s (%d files)", path, len(dir_texts))

        else:
            # Try as HuggingFace dataset
            try:
                from datasets import load_dataset as hf_load

                ds = hf_load(source, split="train")
                # Find text column
                text_col = None
                for col in ["text", "content", "body", "passage", "document"]:
                    if col in ds.column_names:
                        text_col = col
                        break
                if text_col is None:
                    text_col = ds.column_names[0]

                for row in ds:
                    if row[text_col]:
                        all_texts.append(str(row[text_col]))

                logger.info("Loaded HF dataset: %s (%d documents)", source, len(ds))
            except Exception as e:
                logger.warning("⚠️  Could not load '%s': %s", source, e)

    if not all_texts:
        raise ValueError(f"No text loaded from sources: {sources}")

    total_chars = sum(len(t) for t in all_texts)
    logger.info("Total raw text: %s chars from %d documents", f"{total_chars:,}", len(all_texts))

    # ── Chunk the texts ───────────────────────────────────────────────
    chunks = []
    for text in all_texts:
        text_chunks = _chunk_text(text, chunk_size, chunk_overlap, min_chunk_length)
        chunks.extend(text_chunks)

    logger.info("Created %d chunks (avg %d chars each)", len(chunks), total_chars // max(len(chunks), 1))

    # Build dataset
    ds = Dataset.from_dict({"text": chunks})
    return ds


def _load_file(path: Path) -> list[str]:
    """Load a single file and return its text content."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)
    elif suffix in TEXT_EXTENSIONS:
        return [path.read_text(encoding="utf-8", errors="ignore")]
    else:
        # Try reading as text anyway
        try:
            return [path.read_text(encoding="utf-8", errors="ignore")]
        except Exception:
            logger.warning("⚠️  Skipping unreadable file: %s", path)
            return []


def _load_pdf(path: Path) -> list[str]:
    """Extract text from a PDF file."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(path))
        texts = []
        for page in doc:
            text = page.get_text()
            if text.strip():
                texts.append(text)
        doc.close()
        logger.info("Extracted %d pages from PDF: %s", len(texts), path.name)
        return texts
    except ImportError:
        logger.warning(
            "⚠️  PyMuPDF not installed. Install with: pip install PyMuPDF\n"
            "    Trying basic text extraction for: %s", path.name
        )
        # Fallback: try to read as text (won't work for most PDFs)
        try:
            return [path.read_text(encoding="utf-8", errors="ignore")]
        except Exception:
            return []


def _load_directory(path: Path) -> list[str]:
    """Recursively load all text files from a directory."""
    texts = []
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file() and file_path.suffix.lower() in TEXT_EXTENSIONS:
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                if text.strip():
                    texts.append(text)
            except Exception:
                continue
    return texts


def _chunk_text(
    text: str,
    chunk_size: int = 2048,
    overlap: int = 128,
    min_length: int = 50,
) -> list[str]:
    """Split text into overlapping chunks, preferring sentence/paragraph boundaries.

    Strategy:
      1. Try to split at paragraph boundaries (double newline)
      2. Within paragraphs, try sentence boundaries (. ? !)
      3. Fallback to character-level splitting
    """
    if len(text) <= chunk_size:
        return [text] if len(text) >= min_length else []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            chunk = text[start:]
            if len(chunk) >= min_length:
                chunks.append(chunk)
            break

        # Try to find a good break point
        chunk_text = text[start:end]

        # Prefer paragraph boundary
        last_para = chunk_text.rfind("\n\n")
        if last_para > chunk_size * 0.5:
            end = start + last_para + 2
        else:
            # Try sentence boundary
            for sep in [". ", ".\n", "? ", "! ", ";\n"]:
                last_sent = chunk_text.rfind(sep)
                if last_sent > chunk_size * 0.5:
                    end = start + last_sent + len(sep)
                    break

        chunk = text[start:end].strip()
        if len(chunk) >= min_length:
            chunks.append(chunk)

        # Move forward with overlap
        start = end - overlap

    return chunks


def run_cpt(cfg: "TrainingConfig") -> None:
    """Run Continued Pre-Training on raw text data.

    Parameters
    ----------
    cfg:
        TrainingConfig with ``cpt_sources`` set.
    """
    import torch
    from peft import get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    logger.info("═══ Starting Continued Pre-Training (CPT) ═══")

    # ── Load and chunk raw text ───────────────────────────────────────
    logger.info("Loading raw text from: %s", cfg.cpt_sources)
    raw_ds = load_raw_texts(
        sources=cfg.cpt_sources,
        chunk_size=cfg.cpt_chunk_size,
        chunk_overlap=cfg.cpt_chunk_overlap,
    )
    logger.info("CPT dataset: %d chunks", len(raw_ds))

    # ── Load model + tokenizer ────────────────────────────────────────
    logger.info("Loading model: %s", cfg.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto",
    )

    # ── Optionally apply LoRA ─────────────────────────────────────────
    if cfg.use_lora:
        from lfm_trainer.trainer import _build_lora_config

        logger.info("Applying LoRA (r=%d, alpha=%d)", cfg.lora_r, cfg.lora_alpha)
        lora_config = _build_lora_config(cfg)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    # ── Tokenize ──────────────────────────────────────────────────────
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )

    tokenized_ds = raw_ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
        desc="Tokenizing",
    )

    # Train/eval split
    if cfg.eval_split > 0:
        split = tokenized_ds.train_test_split(test_size=cfg.eval_split, seed=42)
        train_ds, eval_ds = split["train"], split["test"]
    else:
        train_ds, eval_ds = tokenized_ds, None

    # ── Data collator for CLM ─────────────────────────────────────────
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )

    # ── Training args ─────────────────────────────────────────────────
    cpt_output_dir = f"{cfg.output_dir}/cpt-adapter"
    os.makedirs(cpt_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=cpt_output_dir,
        num_train_epochs=cfg.cpt_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.cpt_learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        fp16=cfg.fp16 and not cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=2,
        report_to=cfg.report_to if cfg.report_to != "none" else "none",
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=cfg.save_steps if eval_ds else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    # ── Train ─────────────────────────────────────────────────────────
    logger.info(
        "CPT config: lr=%s, epochs=%d, chunks=%d, chunk_size=%d",
        cfg.cpt_learning_rate, cfg.cpt_epochs, len(train_ds), cfg.cpt_chunk_size,
    )
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────
    trainer.save_model(cpt_output_dir)
    tokenizer.save_pretrained(cpt_output_dir)
    logger.info("✅ CPT adapter saved to %s", cpt_output_dir)

    # Push to Hub
    if cfg.push_to_hub and cfg.hub_repo_id:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=cfg.hf_token)
            api.upload_folder(
                folder_path=cpt_output_dir,
                repo_id=cfg.hub_repo_id,
                commit_message="Continued Pre-Training adapter",
            )
            logger.info("✅ CPT adapter pushed to %s", cfg.hub_repo_id)
        except Exception as e:
            logger.warning("⚠️  Push failed: %s", e)

    logger.info("═══ CPT complete ═══")
    return model, tokenizer
