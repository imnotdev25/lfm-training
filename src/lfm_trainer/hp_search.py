"""
Auto hyperparameter search — try a few learning rates and pick the best.

Runs short trials (a few steps each) with different hyperparameter
combinations, evaluates on a held-out split, and returns the best config.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lfm_trainer.config import TrainingConfig

logger = logging.getLogger(__name__)


# Default search space
DEFAULT_SEARCH_SPACE = {
    "learning_rate": [1e-4, 2e-4, 5e-4, 5e-5],
    "lora_r": [8, 16, 32],
    "warmup_ratio": [0.03, 0.06, 0.1],
}


def auto_hp_search(
    cfg: "TrainingConfig",
    search_space: dict | None = None,
    trial_steps: int = 50,
    eval_split: float = 0.1,
) -> "TrainingConfig":
    """Run a quick hyperparameter search and return the best config.

    Parameters
    ----------
    cfg:
        Base training config.
    search_space:
        Dict mapping param names to lists of values to try.
        Default: tries 4 learning rates, 3 LoRA ranks, 3 warmup ratios.
    trial_steps:
        Number of training steps per trial (default: 50). Lower = faster.
    eval_split:
        Fraction of data for evaluation (overrides cfg.eval_split).

    Returns
    -------
    TrainingConfig with the best hyperparameters set.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lfm_trainer.data import load_datasets
    from lfm_trainer.trainer import _build_lora_config

    if search_space is None:
        search_space = {"learning_rate": DEFAULT_SEARCH_SPACE["learning_rate"]}

    # Build trial combinations
    trials = _build_trials(search_space)
    logger.info("═══ Auto HP Search: %d trials, %d steps each ═══", len(trials), trial_steps)

    # Load data once (with eval split)
    logger.info("Loading dataset for HP search…")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name, trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    result = load_datasets(
        cfg.dataset_paths,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        text_column=cfg.dataset_text_column,
        tool_calling_only=cfg.tool_calling_only,
        quality_filter=True,
        eval_split=max(eval_split, 0.1),
    )

    if isinstance(result, tuple):
        train_ds, eval_ds = result
    else:
        # Force a split
        split = result.train_test_split(test_size=0.1, seed=42)
        train_ds, eval_ds = split["train"], split["test"]

    logger.info("Search data: %d train, %d eval examples", len(train_ds), len(eval_ds))

    # Run trials
    best_loss = float("inf")
    best_params = {}
    results = []

    for i, params in enumerate(trials):
        logger.info("─── Trial %d/%d: %s ───", i + 1, len(trials), params)

        try:
            eval_loss = _run_trial(
                cfg=cfg,
                params=params,
                train_ds=train_ds,
                eval_ds=eval_ds,
                tokenizer=tokenizer,
                trial_steps=trial_steps,
            )

            results.append({"params": params, "eval_loss": eval_loss})
            logger.info("  eval_loss = %.4f", eval_loss)

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_params = params
                logger.info("  🏆 New best!")

        except Exception as e:
            logger.warning("  ❌ Trial failed: %s", e)
            results.append({"params": params, "eval_loss": float("inf"), "error": str(e)})

        # Clean up GPU memory between trials
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Report
    logger.info("")
    logger.info("═══ HP Search Results ═══")
    for r in sorted(results, key=lambda x: x["eval_loss"]):
        marker = "🏆" if r["params"] == best_params else "  "
        loss_str = f"{r['eval_loss']:.4f}" if r["eval_loss"] < float("inf") else "FAILED"
        logger.info("%s %s → eval_loss=%s", marker, r["params"], loss_str)

    logger.info("")
    logger.info("Best: %s (eval_loss=%.4f)", best_params, best_loss)

    # Create updated config
    best_cfg = replace(cfg, **best_params, eval_split=eval_split)
    return best_cfg


def _run_trial(
    cfg: "TrainingConfig",
    params: dict,
    train_ds,
    eval_ds,
    tokenizer,
    trial_steps: int,
) -> float:
    """Run a single HP trial and return eval loss."""
    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    from lfm_trainer.trainer import _build_lora_config

    # Create trial config
    trial_cfg = replace(cfg, **params)

    # Fresh model for each trial
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
        torch_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
        device_map="auto",
    )

    if cfg.use_lora:
        lora_config = _build_lora_config(trial_cfg)
        model = get_peft_model(model, lora_config)

    training_args = SFTConfig(
        output_dir=f"/tmp/hp_trial_{id(params)}",
        max_steps=trial_steps,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=params.get("learning_rate", cfg.learning_rate),
        warmup_ratio=params.get("warmup_ratio", cfg.warmup_ratio),
        max_seq_length=cfg.max_seq_length,
        bf16=cfg.bf16,
        fp16=cfg.fp16 and not cfg.bf16,
        logging_steps=trial_steps,
        eval_strategy="steps",
        eval_steps=trial_steps,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    # Get eval loss
    eval_result = trainer.evaluate()
    eval_loss = eval_result.get("eval_loss", float("inf"))

    # Cleanup
    del model, trainer
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return eval_loss


def _build_trials(search_space: dict) -> list[dict]:
    """Build trial combinations from search space.

    For efficiency, we do a grid search on single params or a limited
    combination for multi-param searches (max 12 trials).
    """
    import itertools

    keys = list(search_space.keys())
    values = list(search_space.values())

    # Full grid
    trials = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    # Cap at 12 trials to keep things reasonable
    if len(trials) > 12:
        logger.info("Capping %d trial combos to 12 (most diverse)", len(trials))
        # Take evenly spaced samples
        step = len(trials) / 12
        trials = [trials[int(i * step)] for i in range(12)]

    return trials
