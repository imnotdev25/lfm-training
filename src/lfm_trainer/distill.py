"""
Model Distillation — transfer knowledge from a large teacher to a smaller student.

The student learns from the teacher's output probability distribution (soft labels),
which carries more information than hard labels alone:
- Relative probabilities between wrong answers encode knowledge
- Temperature-scaled softmax smooths the distribution for better transfer

Loss function:
    L = α * KL(student_soft || teacher_soft) + (1-α) * CE(student_logits, labels)

Where:
    - teacher_soft = softmax(teacher_logits / T)
    - student_soft = softmax(student_logits / T)
    - T = temperature (higher = softer distribution)
    - α = blend factor (0 = pure CE, 1 = pure KL)
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class DistillationTrainer(Trainer):
    """Custom Trainer that computes a blended KL-divergence + CE loss."""

    def __init__(
        self,
        teacher_model: AutoModelForCausalLM,
        temperature: float = 2.0,
        alpha: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher = teacher_model
        self.teacher.eval()
        self.temperature = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute blended distillation loss.

        L = α * T² * KL(student_soft, teacher_soft) + (1-α) * CE(logits, labels)
        """
        labels = inputs.get("labels")

        # Student forward
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Teacher forward (no gradients)
        with torch.no_grad():
            teacher_outputs = self.teacher(**inputs)
            teacher_logits = teacher_outputs.logits

        # ── KL divergence on temperature-scaled distributions ─────────
        T = self.temperature
        student_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(teacher_logits / T, dim=-1)

        # KL(student || teacher), scaled by T² (standard practice)
        kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T * T)

        # ── Standard cross-entropy loss ───────────────────────────────
        if labels is not None:
            # Shift for causal LM: predict next token
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        else:
            ce_loss = torch.tensor(0.0, device=student_logits.device)

        # ── Blended loss ──────────────────────────────────────────────
        loss = self.alpha * kl_loss + (1.0 - self.alpha) * ce_loss

        return (loss, student_outputs) if return_outputs else loss


def run_distillation(cfg) -> None:
    """Run knowledge distillation from teacher → student.

    Parameters
    ----------
    cfg : TrainingConfig
        Must have ``distill_teacher`` set to a HuggingFace model ID.
        The ``model_name`` field is the student model.
    """
    from lfm_trainer.data import load_datasets

    logger.info("Loading teacher model: %s", cfg.distill_teacher)
    logger.info("Loading student model: %s", cfg.model_name)

    dtype = torch.float16 if cfg.fp16 else (torch.bfloat16 if cfg.bf16 else torch.float32)

    # ── Load tokenizer (from teacher — larger vocab) ──────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.distill_teacher,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load teacher (frozen) ─────────────────────────────────────────
    teacher = AutoModelForCausalLM.from_pretrained(
        cfg.distill_teacher,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=dtype,
        device_map="auto",
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_params = sum(p.numel() for p in teacher.parameters())
    logger.info("Teacher: %s (%.1fB params, frozen)", cfg.distill_teacher, teacher_params / 1e9)

    # ── Load student (trainable) ──────────────────────────────────────
    student = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=dtype,
        device_map="auto",
    )

    # Resize student embeddings to match teacher tokenizer
    student.resize_token_embeddings(len(tokenizer))

    student_params = sum(p.numel() for p in student.parameters())
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    logger.info("Student: %s (%.1fB params, %.1fB trainable)", cfg.model_name, student_params / 1e9, trainable / 1e9)
    logger.info("Compression ratio: %.1fx", teacher_params / student_params)

    # ── Optional: LoRA on student ─────────────────────────────────────
    if cfg.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        student = get_peft_model(student, lora_config)
        student.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────
    logger.info("Loading datasets: %s", cfg.dataset_paths)
    result = load_datasets(
        cfg.dataset_paths,
        text_column=cfg.dataset_text_column,
        tool_calling_only=cfg.tool_calling_only,
        quality_filter=cfg.quality_filter,
        eval_split=cfg.eval_split,
    )
    if isinstance(result, tuple):
        train_dataset, eval_dataset = result
    else:
        train_dataset, eval_dataset = result, None

    # ── Training args ─────────────────────────────────────────────────
    training_args = TrainingArguments(
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
        report_to=cfg.report_to,
    )

    # ── Distillation trainer ──────────────────────────────────────────
    trainer_kwargs = dict(
        teacher_model=teacher,
        temperature=cfg.distill_temperature,
        alpha=cfg.distill_alpha,
        model=student,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
        training_args.eval_strategy = "steps"
        training_args.eval_steps = cfg.save_steps

    trainer = DistillationTrainer(**trainer_kwargs)

    logger.info(
        "Starting distillation: T=%.1f, α=%.2f (%.0f%% KL + %.0f%% CE)",
        cfg.distill_temperature,
        cfg.distill_alpha,
        cfg.distill_alpha * 100,
        (1 - cfg.distill_alpha) * 100,
    )
    trainer.train()

    # ── Save ──────────────────────────────────────────────────────────
    save_path = f"{cfg.output_dir}/distilled-model"
    student.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    logger.info("Distilled student saved to %s", save_path)

    # Push to Hub if configured
    if cfg.push_to_hub and cfg.hub_repo_id and cfg.hf_token:
        logger.info("Pushing distilled model to %s", cfg.hub_repo_id)
        student.push_to_hub(cfg.hub_repo_id, token=cfg.hf_token)
        tokenizer.push_to_hub(cfg.hub_repo_id, token=cfg.hf_token)
        logger.info("✅ Distilled model published to Hub")
