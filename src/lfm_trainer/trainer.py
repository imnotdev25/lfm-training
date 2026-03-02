"""
Training orchestrator — wires model, dataset, LoRA, and trainer together.
"""

from __future__ import annotations

import logging

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTConfig, SFTTrainer

from lfm_trainer.callbacks import safe_train
from lfm_trainer.config import TrainingConfig
from lfm_trainer.data import load_datasets
from lfm_trainer.export import run_exports

logger = logging.getLogger(__name__)


def _build_lora_config(cfg: TrainingConfig) -> LoraConfig:
    return LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def run_training(cfg: TrainingConfig) -> None:
    """End-to-end training pipeline.

    1. Load model & tokenizer
    2. Attach LoRA adapters
    3. Load & merge datasets
    4. Configure SFTTrainer
    5. Train with error-resilient wrapper
    6. Export GGUF / MLX quantized versions (if enabled)
    """
    # ── 1. Model & tokenizer ──────────────────────────────────────────
    logger.info("Loading model: %s", cfg.model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Ensure LFM 2.5 tool-calling special tokens are registered
    TOOL_TOKENS = [
        "<|tool_call_start|>",
        "<|tool_call_end|>",
        "<|tool_result_start|>",
        "<|tool_result_end|>",
    ]
    tokens_to_add = [t for t in TOOL_TOKENS if t not in tokenizer.get_vocab()]
    if tokens_to_add:
        logger.info("Adding %d tool-calling special tokens: %s", len(tokens_to_add), tokens_to_add)
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch.float16 if cfg.fp16 else (torch.bfloat16 if cfg.bf16 else torch.float32),
        device_map="auto",
    )

    # Resize embeddings if new tokens were added
    if tokens_to_add:
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Resized model embeddings to %d", len(tokenizer))

    # ── 2. LoRA adapters ──────────────────────────────────────────────
    if cfg.resume_from_model:
        # Continual training: load a previously trained adapter
        logger.info("Resuming from prior adapter: %s", cfg.resume_from_model)
        model = PeftModel.from_pretrained(
            model,
            cfg.resume_from_model,
            is_trainable=True,
        )
        logger.info("Loaded existing adapter — continuing training")
    else:
        # Fresh LoRA
        logger.info("Applying fresh LoRA (r=%d, alpha=%d)", cfg.lora_r, cfg.lora_alpha)
        lora_config = _build_lora_config(cfg)
        model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── 3. Dataset ────────────────────────────────────────────────────
    logger.info("Loading datasets: %s", cfg.dataset_paths)
    dataset = load_datasets(
        cfg.dataset_paths,
        text_column=cfg.dataset_text_column,
        tool_calling_only=cfg.tool_calling_only,
    )

    # ── 4. Trainer ────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_strategy=cfg.save_strategy,
        save_total_limit=cfg.save_total_limit,
        max_length=cfg.max_seq_length,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # ── 5. Train with error handling ──────────────────────────────────
    repo_id = cfg.hub_repo_id or "lfm2.5-1.2b-code-finetune"
    safe_train(
        trainer=trainer,
        model=model,
        tokenizer=tokenizer,
        repo_id=repo_id,
        token=cfg.hf_token,
        simulate_error=cfg.simulate_error,
        push_to_hub=cfg.push_to_hub,
        output_dir=cfg.output_dir,
    )

    # ── 6. Post-training export (GGUF / MLX) ──────────────────────────
    if cfg.export_gguf or cfg.export_mlx:
        from lfm_trainer.callbacks import _version_tag

        version_tag = _version_tag()
        logger.info("Starting post-training export (version: %s)", version_tag)

        # Merge LoRA into base model and save locally for export
        merged_dir = f"{cfg.output_dir}/merged-for-export"
        logger.info("Merging LoRA adapters into base model → %s", merged_dir)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        run_exports(
            model_dir=merged_dir,
            repo_id_base=repo_id,
            version_tag=version_tag,
            token=cfg.hf_token,
            output_base=cfg.export_output_dir,
            enable_gguf=cfg.export_gguf,
            enable_mlx=cfg.export_mlx,
        )

