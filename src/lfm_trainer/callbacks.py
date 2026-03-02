"""
Automatic checkpoint saving and Hugging Face Hub publishing on errors.

Inspired by Unsloth's error-resilient training pattern:
- Catches OOM, SIGTERM (Kaggle timeout), KeyboardInterrupt, and generic exceptions.
- Saves the current adapter / merged model to the Hub with a versioned tag.
"""

from __future__ import annotations

import gc
import logging
import signal
import sys
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Optional

import torch
from huggingface_hub import HfApi

if TYPE_CHECKING:
    from peft import PeftModel
    from transformers import PreTrainedTokenizerBase, Trainer

logger = logging.getLogger(__name__)


def _version_tag() -> str:
    """Generate a unique version tag from the current UTC timestamp."""
    return datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M%S")


def _safe_push_to_hub(
    trainer: Trainer,
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    repo_id: str,
    token: Optional[str],
    reason: str,
) -> None:
    """Best-effort push of the current model state to the Hugging Face Hub."""
    tag = _version_tag()
    logger.warning(
        "⚠️  Auto-publishing checkpoint to Hub (reason: %s, tag: %s)", reason, tag
    )

    try:
        # Save locally first
        save_dir = f"./lfm-emergency-{tag}"
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Push adapter weights
        model.push_to_hub(repo_id, token=token, commit_message=f"[auto] {reason} — {tag}")
        tokenizer.push_to_hub(repo_id, token=token, commit_message=f"[auto] {reason} — {tag}")

        # Tag the commit for easy identification
        api = HfApi(token=token)
        api.create_tag(repo_id=repo_id, tag=tag, tag_message=f"Auto-checkpoint: {reason}")

        logger.info("✅ Successfully pushed checkpoint %s to %s", tag, repo_id)

    except Exception as push_err:
        logger.error("❌ Failed to push checkpoint to Hub: %s", push_err)
        logger.info("💾 Local emergency save is at: %s", save_dir)


# ── SIGTERM handler (Kaggle session timeout) ───────────────────────────────

# We store references so the signal handler can access them.
_GLOBAL_REFS: dict = {}


def _sigterm_handler(signum: int, frame, **_kwargs) -> None:  # noqa: ANN001
    """Handle SIGTERM gracefully — save and push, then exit."""
    logger.warning("🛑 Received SIGTERM (signal %d) — Kaggle session likely ending.", signum)
    refs = _GLOBAL_REFS
    if refs:
        _safe_push_to_hub(
            trainer=refs["trainer"],
            model=refs["model"],
            tokenizer=refs["tokenizer"],
            repo_id=refs["repo_id"],
            token=refs["token"],
            reason="SIGTERM",
        )
    sys.exit(0)


def register_sigterm_handler(
    trainer: Trainer,
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    repo_id: str,
    token: Optional[str],
) -> None:
    """Register SIGTERM handler with references to current training objects."""
    _GLOBAL_REFS.update(
        {
            "trainer": trainer,
            "model": model,
            "tokenizer": tokenizer,
            "repo_id": repo_id,
            "token": token,
        }
    )
    signal.signal(signal.SIGTERM, _sigterm_handler)
    logger.info("Registered SIGTERM handler for auto-checkpoint.")


# ── Main wrapper ───────────────────────────────────────────────────────────

def safe_train(
    trainer: Trainer,
    model: PeftModel,
    tokenizer: PreTrainedTokenizerBase,
    repo_id: str,
    token: Optional[str],
    simulate_error: bool = False,
    push_to_hub: bool = True,
    output_dir: str = "./lfm-checkpoints",
) -> None:
    """Run ``trainer.train()`` inside a resilient error-handling wrapper.

    On any failure the current model state is pushed to the Hub with a
    versioned tag so that no work is lost.

    When ``push_to_hub=False``, the model is only saved locally on success
    (useful for intermediate continual-training rounds).
    """
    # Register the SIGTERM handler so Kaggle timeouts are caught
    register_sigterm_handler(trainer, model, tokenizer, repo_id, token)

    try:
        if simulate_error:
            # Train for a few steps, then simulate an error
            logger.warning("🧪 --simulate-error is ON — will raise after 5 steps")
            trainer.args.max_steps = 5
            trainer.train()
            raise RuntimeError("Simulated training error for testing auto-publish")

        trainer.train()
        logger.info("🎉 Training completed successfully!")

        if push_to_hub:
            # Push final model to Hub
            model.push_to_hub(repo_id, token=token, commit_message="Training complete")
            tokenizer.push_to_hub(repo_id, token=token, commit_message="Training complete")
            logger.info("✅ Final model pushed to %s", repo_id)
        else:
            # Save locally only
            save_dir = f"{output_dir}/final-adapter"
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            logger.info("💾 Model saved locally to %s (push_to_hub=False)", save_dir)

    except torch.cuda.OutOfMemoryError:
        logger.error("💥 CUDA Out-of-Memory! Attempting emergency checkpoint…")
        torch.cuda.empty_cache()
        gc.collect()
        _safe_push_to_hub(trainer, model, tokenizer, repo_id, token, reason="OOM")

    except KeyboardInterrupt:
        logger.warning("⏹️  KeyboardInterrupt — saving checkpoint…")
        _safe_push_to_hub(trainer, model, tokenizer, repo_id, token, reason="KeyboardInterrupt")

    except Exception as exc:
        logger.error("💥 Unexpected error: %s", exc)
        _safe_push_to_hub(trainer, model, tokenizer, repo_id, token, reason=f"Exception: {exc!r}")
        raise  # Re-raise so the user sees the traceback
