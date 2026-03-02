"""
Alignment module — DPO, PPO, GRPO, and RLHF training stages.

After SFT, run one of these to align the model with human preferences:
  - DPO:  Direct Preference Optimization (simple, stable, recommended)
  - PPO:  Proximal Policy Optimization (classic RLHF with reward model)
  - GRPO: Group Relative Policy Optimization (DeepSeek's reward-function approach)

Each method requires different data:
  - DPO:  preference pairs (chosen/rejected)
  - PPO:  prompts + reward model (or reward function)
  - GRPO: prompts + reward function
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Callable, Optional

import torch
from peft import LoraConfig, PeftModel

if TYPE_CHECKING:
    from lfm_trainer.config import TrainingConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
#  Dispatcher
# ═══════════════════════════════════════════════════════════════════════

def run_alignment(
    cfg: "TrainingConfig",
    model=None,
    tokenizer=None,
    reward_fn: Callable | None = None,
):
    """Run the selected alignment method after SFT.

    Parameters
    ----------
    cfg:
        Training config. ``cfg.alignment_method`` selects DPO/PPO/GRPO.
    model:
        SFT model to align. If None, loads from output_dir.
    tokenizer:
        Tokenizer. If None, loads from output_dir.
    reward_fn:
        Custom reward function (required for GRPO, optional for PPO).
        Signature: ``reward_fn(completions: list[str], **kwargs) -> list[float]``
    """
    method = cfg.alignment_method.lower()
    logger.info("═══ Starting %s alignment stage ═══", method.upper())

    if method == "dpo":
        return run_dpo(cfg, model=model, tokenizer=tokenizer)
    elif method == "ppo":
        return run_ppo(cfg, model=model, tokenizer=tokenizer, reward_fn=reward_fn)
    elif method == "grpo":
        return run_grpo(cfg, model=model, tokenizer=tokenizer, reward_fn=reward_fn)
    else:
        raise ValueError(
            f"Unknown alignment method: '{method}'. "
            f"Supported: dpo, ppo, grpo"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Shared Helpers
# ═══════════════════════════════════════════════════════════════════════

def _load_model_and_tokenizer(cfg: "TrainingConfig", model, tokenizer):
    """Load model/tokenizer if not provided."""
    if model is not None and tokenizer is not None:
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    sft_adapter_path = cfg.alignment_sft_model or f"{cfg.output_dir}/final-adapter"
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
            base_model, sft_adapter_path, is_trainable=True,
        )
    else:
        model = base_model

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def _find_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    """Find a column name from a list of candidates."""
    for c in candidates:
        if c in columns:
            return c
    return None


def _save_and_push(cfg, output_dir, tokenizer, commit_msg: str):
    """Save tokenizer and push to Hub if configured."""
    tokenizer.save_pretrained(output_dir)

    if cfg.push_to_hub and cfg.hub_repo_id:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=cfg.hf_token)
            api.upload_folder(
                folder_path=output_dir,
                repo_id=cfg.hub_repo_id,
                commit_message=commit_msg,
            )
            logger.info("✅ Pushed to %s", cfg.hub_repo_id)
        except Exception as e:
            logger.warning("⚠️  Push failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════
#  DPO — Direct Preference Optimization
# ═══════════════════════════════════════════════════════════════════════

def run_dpo(cfg: "TrainingConfig", model=None, tokenizer=None):
    """DPO alignment using preference pairs (chosen vs rejected).

    Dataset format: prompt, chosen, rejected columns.
    """
    from datasets import load_dataset
    from trl import DPOConfig, DPOTrainer

    model, tokenizer = _load_model_and_tokenizer(cfg, model, tokenizer)

    # Load preference dataset
    logger.info("Loading DPO dataset: %s", cfg.alignment_dataset)
    dpo_ds = load_dataset(cfg.alignment_dataset, split="train")

    # Auto-detect column mapping
    columns = dpo_ds.column_names
    prompt_col = _find_column(columns, ["prompt", "question", "instruction", "input"])
    chosen_col = _find_column(columns, ["chosen", "preferred", "accepted", "positive"])
    rejected_col = _find_column(columns, ["rejected", "dispreferred", "negative", "refused"])

    if not all([prompt_col, chosen_col, rejected_col]):
        raise ValueError(
            f"DPO dataset must have prompt, chosen, and rejected columns. "
            f"Found: {columns}"
        )

    if prompt_col != "prompt":
        dpo_ds = dpo_ds.rename_column(prompt_col, "prompt")
    if chosen_col != "chosen":
        dpo_ds = dpo_ds.rename_column(chosen_col, "chosen")
    if rejected_col != "rejected":
        dpo_ds = dpo_ds.rename_column(rejected_col, "rejected")

    logger.info("DPO dataset: %d examples", len(dpo_ds))

    output_dir = f"{cfg.output_dir}/dpo-adapter"
    os.makedirs(output_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.alignment_epochs,
        per_device_train_batch_size=cfg.alignment_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.alignment_learning_rate,
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

    logger.info("DPO: β=%.2f, lr=%s, epochs=%d", cfg.dpo_beta, cfg.alignment_learning_rate, cfg.alignment_epochs)
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("✅ DPO adapter saved to %s", output_dir)

    _save_and_push(cfg, output_dir, tokenizer, "DPO alignment adapter")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
#  PPO — Proximal Policy Optimization (Classic RLHF)
# ═══════════════════════════════════════════════════════════════════════

def run_ppo(cfg: "TrainingConfig", model=None, tokenizer=None, reward_fn=None):
    """PPO alignment using a reward model or reward function.

    This is the classic RLHF approach: generate → score → update.

    Dataset format: prompts (single column).
    If no reward_fn is provided, uses a reward model from ``cfg.reward_model``.
    """
    from datasets import load_dataset
    from trl import PPOConfig, PPOTrainer

    model, tokenizer = _load_model_and_tokenizer(cfg, model, tokenizer)

    # Load prompts dataset
    logger.info("Loading PPO prompt dataset: %s", cfg.alignment_dataset)
    ppo_ds = load_dataset(cfg.alignment_dataset, split="train")

    # Auto-detect prompt column
    columns = ppo_ds.column_names
    prompt_col = _find_column(columns, ["prompt", "query", "question", "instruction", "input", "text"])
    if prompt_col and prompt_col != "query":
        ppo_ds = ppo_ds.rename_column(prompt_col, "query")

    output_dir = f"{cfg.output_dir}/ppo-adapter"
    os.makedirs(output_dir, exist_ok=True)

    ppo_config = PPOConfig(
        output_dir=output_dir,
        learning_rate=cfg.alignment_learning_rate,
        batch_size=cfg.alignment_batch_size * cfg.gradient_accumulation_steps,
        mini_batch_size=cfg.alignment_batch_size,
        ppo_epochs=cfg.ppo_ppo_epochs,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        log_with=cfg.report_to if cfg.report_to != "none" else None,
    )

    # Build reward model or function
    reward_model = None
    if reward_fn is None and cfg.reward_model:
        from transformers import AutoModelForSequenceClassification

        logger.info("Loading reward model: %s", cfg.reward_model)
        reward_model = AutoModelForSequenceClassification.from_pretrained(
            cfg.reward_model,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            device_map="auto",
        )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        tokenizer=tokenizer,
        dataset=ppo_ds,
    )

    logger.info(
        "PPO: lr=%s, batch=%d, ppo_epochs=%d, steps=%d",
        cfg.alignment_learning_rate, ppo_config.batch_size,
        cfg.ppo_ppo_epochs, cfg.alignment_max_steps,
    )

    # Training loop
    for step, batch in enumerate(trainer.dataloader):
        if step >= cfg.alignment_max_steps:
            break

        # Generate responses
        query_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze() for q in batch["query"]]
        response_tensors = trainer.generate(query_tensors, max_new_tokens=cfg.alignment_max_new_tokens)
        responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

        # Score responses
        if reward_fn:
            rewards = reward_fn(responses, prompts=batch["query"])
            rewards = [torch.tensor(r, dtype=torch.float32) for r in rewards]
        elif reward_model:
            rewards = []
            for query, response in zip(batch["query"], responses):
                inputs = tokenizer(query + response, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(reward_model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    score = reward_model(**inputs).logits[0, 0]
                rewards.append(score.cpu())
        else:
            raise ValueError("PPO requires either a reward_fn or cfg.reward_model")

        # PPO update
        stats = trainer.step(query_tensors, response_tensors, rewards)

        if step % cfg.logging_steps == 0:
            mean_reward = sum(r.item() for r in rewards) / len(rewards)
            logger.info("PPO step %d/%d: mean_reward=%.3f", step, cfg.alignment_max_steps, mean_reward)

    # Save
    trainer.save_pretrained(output_dir)
    logger.info("✅ PPO adapter saved to %s", output_dir)

    _save_and_push(cfg, output_dir, tokenizer, "PPO RLHF alignment")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
#  GRPO — Group Relative Policy Optimization (DeepSeek)
# ═══════════════════════════════════════════════════════════════════════

def run_grpo(cfg: "TrainingConfig", model=None, tokenizer=None, reward_fn=None):
    """GRPO alignment using reward functions (no reward model needed).

    GRPO generates multiple completions per prompt, scores them with a
    reward function, and uses the group's relative ranking to update.
    This is simpler than PPO and doesn't need a separate reward model.

    Dataset format: prompts with a "prompt" column.
    """
    from datasets import load_dataset
    from trl import GRPOConfig, GRPOTrainer

    # Use default reward function if none provided
    if reward_fn is None:
        reward_fn = _default_code_reward_fn

    # Load prompts dataset
    logger.info("Loading GRPO prompt dataset: %s", cfg.alignment_dataset)
    grpo_ds = load_dataset(cfg.alignment_dataset, split="train")

    # Auto-detect and rename prompt column
    columns = grpo_ds.column_names
    prompt_col = _find_column(columns, ["prompt", "query", "question", "instruction", "input", "text"])
    if prompt_col and prompt_col != "prompt":
        grpo_ds = grpo_ds.rename_column(prompt_col, "prompt")

    output_dir = f"{cfg.output_dir}/grpo-adapter"
    os.makedirs(output_dir, exist_ok=True)

    # Build LoRA config if using LoRA
    peft_config = None
    if cfg.use_lora:
        peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
        )

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=cfg.alignment_epochs,
        per_device_train_batch_size=cfg.alignment_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.alignment_learning_rate,
        num_generations=cfg.grpo_num_generations,
        max_completion_length=cfg.alignment_max_new_tokens,
        bf16=cfg.bf16,
        fp16=cfg.fp16 and not cfg.bf16,
        gradient_checkpointing=True,
        logging_steps=cfg.logging_steps,
        save_strategy="no",
        report_to=cfg.report_to if cfg.report_to != "none" else "none",
    )

    # For GRPO, pass model name (not loaded model) when using LoRA
    model_ref = cfg.model_name if model is None else model

    trainer = GRPOTrainer(
        model=model_ref,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=grpo_ds,
        peft_config=peft_config,
    )

    logger.info(
        "GRPO: lr=%s, epochs=%d, generations=%d",
        cfg.alignment_learning_rate, cfg.alignment_epochs, cfg.grpo_num_generations,
    )
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("✅ GRPO adapter saved to %s", output_dir)

    _save_and_push(cfg, output_dir, tokenizer or trainer.processing_class, "GRPO alignment")
    return model, tokenizer


# ═══════════════════════════════════════════════════════════════════════
#  Built-in Reward Functions
# ═══════════════════════════════════════════════════════════════════════

def _default_code_reward_fn(completions: list[str], **kwargs) -> list[float]:
    """Built-in reward function for coding tasks.

    Scores based on code quality heuristics:
      - Has function definition (+0.3)
      - Has docstring (+0.2)
      - Has type hints (+0.1)
      - Has reasonable length (+0.1–0.3)
      - Penalize empty or very short (-0.5)
    """
    rewards = []
    for completion in completions:
        score = 0.0

        if not completion or len(completion.strip()) < 10:
            rewards.append(-0.5)
            continue

        # Structural quality signals
        if "def " in completion or "class " in completion:
            score += 0.3
        if '"""' in completion or "'''" in completion or "# " in completion:
            score += 0.2
        if "->" in completion or ": " in completion:
            score += 0.1

        # Length reward (prefer moderate length)
        length = len(completion)
        if 50 < length < 500:
            score += 0.3
        elif 20 < length <= 50:
            score += 0.1
        elif length >= 500:
            score += 0.1  # still okay, just long

        # Penalize repetition
        lines = completion.strip().split("\n")
        unique_lines = set(lines)
        if len(lines) > 3 and len(unique_lines) / len(lines) < 0.5:
            score -= 0.3

        rewards.append(score)

    return rewards


def code_correctness_reward(completions: list[str], **kwargs) -> list[float]:
    """Reward function that checks if generated code is syntactically valid.

    Returns 1.0 for valid Python, 0.0 for syntax errors.
    This is a simple but effective reward signal.
    """
    import ast

    rewards = []
    for completion in completions:
        try:
            ast.parse(completion)
            rewards.append(1.0)
        except SyntaxError:
            rewards.append(0.0)
    return rewards


def length_and_quality_reward(completions: list[str], **kwargs) -> list[float]:
    """Combined length and quality reward.

    Rewards concise, well-structured responses.
    """
    rewards = []
    for completion in completions:
        score = 0.0
        length = len(completion)

        # Length: prefer 100-300 chars
        if 100 <= length <= 300:
            score += 0.5
        elif 50 <= length < 100 or 300 < length <= 600:
            score += 0.3
        elif length < 50:
            score -= 0.2

        # Quality signals
        if "\n" in completion:
            score += 0.2  # structured
        if completion.strip().endswith((".", "```", ")")):
            score += 0.1  # complete

        rewards.append(score)
    return rewards
