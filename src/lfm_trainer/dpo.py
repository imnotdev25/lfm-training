"""
DPO (Direct Preference Optimization) — preference alignment stage.

After SFT, run a DPO pass to align the model with human preferences
(e.g., prefer helpful, safe, well-formatted responses over bad ones).

Requires a preference dataset with "chosen" and "rejected" responses.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import torch
from peft import PeftModel

if TYPE_CHECKING:
    from lfm_trainer.config import TrainingConfig

logger = logging.getLogger(__name__)


def run_dpo(
    cfg: "TrainingConfig",
    model=None,
    tokenizer=None,
) -> None:
    """Run DPO training after SFT.

    Parameters
    ----------
    cfg:
        Training config. Must have ``dpo_dataset`` set.
    model:
        Model to align. If None, loads from ``cfg.output_dir/final-adapter``.
    tokenizer:
        Tokenizer. If None, loads from ``cfg.output_dir/final-adapter``.
    """
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    logger.info("═══ Starting DPO alignment stage ═══")

    # ── Load model if not provided ────────────────────────────────────
    if model is None or tokenizer is None:
        sft_adapter_path = cfg.dpo_sft_model or f"{cfg.output_dir}/final-adapter"
        logger.info("Loading SFT model from: %s", sft_adapter_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            sft_adapter_path,
            trust_remote_code=cfg.trust_remote_code,
        )

        if cfg.use_lora:
            model = PeftModel.from_pretrained(
                base_model,
                sft_adapter_path,
                is_trainable=True,
            )
        else:
            model = base_model

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load preference dataset ───────────────────────────────────────
    logger.info("Loading DPO dataset: %s", cfg.dpo_dataset)
    dpo_ds = load_dataset(cfg.dpo_dataset, split="train")

    # Auto-detect column mapping
    columns = dpo_ds.column_names
    prompt_col = _find_column(columns, ["prompt", "question", "instruction", "input"])
    chosen_col = _find_column(columns, ["chosen", "preferred", "accepted", "positive"])
    rejected_col = _find_column(columns, ["rejected", "dispreferred", "negative", "refused"])

    if not all([prompt_col, chosen_col, rejected_col]):
        raise ValueError(
            f"DPO dataset must have prompt, chosen, and rejected columns. "
            f"Found columns: {columns}. "
            f"Expected: prompt/chosen/rejected (or similar)."
        )

    # Rename columns to standard names if needed
    if prompt_col != "prompt":
        dpo_ds = dpo_ds.rename_column(prompt_col, "prompt")
    if chosen_col != "chosen":
        dpo_ds = dpo_ds.rename_column(chosen_col, "chosen")
    if rejected_col != "rejected":
        dpo_ds = dpo_ds.rename_column(rejected_col, "rejected")

    logger.info("DPO dataset: %d examples", len(dpo_ds))

    # ── Configure DPO trainer ─────────────────────────────────────────
    dpo_output_dir = f"{cfg.output_dir}/dpo-adapter"
    os.makedirs(dpo_output_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=dpo_output_dir,
        num_train_epochs=cfg.dpo_epochs,
        per_device_train_batch_size=cfg.dpo_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.dpo_learning_rate,
        beta=cfg.dpo_beta,
        max_length=cfg.max_seq_length,
        max_prompt_length=cfg.max_seq_length // 2,
        bf16=cfg.bf16,
        fp16=cfg.fp16 and not cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        report_to=cfg.report_to if cfg.report_to != "none" else "none",
        remove_unused_columns=False,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dpo_ds,
        processing_class=tokenizer,
    )

    # ── Train ─────────────────────────────────────────────────────────
    logger.info(
        "DPO config: β=%.2f, lr=%s, epochs=%d, batch=%d",
        cfg.dpo_beta, cfg.dpo_learning_rate, cfg.dpo_epochs, cfg.dpo_batch_size,
    )
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────
    trainer.save_model(dpo_output_dir)
    tokenizer.save_pretrained(dpo_output_dir)
    logger.info("✅ DPO adapter saved to %s", dpo_output_dir)

    # Push to Hub if configured
    if cfg.push_to_hub and cfg.hub_repo_id:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=cfg.hf_token)
            api.upload_folder(
                folder_path=dpo_output_dir,
                repo_id=cfg.hub_repo_id,
                commit_message="DPO alignment adapter",
            )
            logger.info("✅ DPO adapter pushed to %s", cfg.hub_repo_id)
        except Exception as e:
            logger.warning("⚠️  DPO push failed: %s", e)

    return model, tokenizer


def _find_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Find a column name from a list of candidates."""
    for c in candidates:
        if c in columns:
            return c
    return None
