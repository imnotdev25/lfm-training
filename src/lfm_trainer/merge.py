"""
Merge multiple LoRA adapters into a single model.

Use cases:
  - Combine round 1/2/3 adapters into one before export
  - Merge adapters trained on different datasets
  - Stack specializations (coding + tool-calling + domain)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def merge_adapters(
    base_model_name: str,
    adapter_paths: list[str],
    output_dir: str,
    weights: list[float] | None = None,
    trust_remote_code: bool = True,
    torch_dtype: str = "float16",
    push_to_hub: bool = False,
    hub_repo_id: str | None = None,
    hf_token: str | None = None,
) -> str:
    """Merge multiple LoRA adapters sequentially into a single model.

    Adapters are applied in order. Each adapter's weights are merged into the
    base model before the next adapter is loaded. This is equivalent to
    continual fine-tuning but without retraining.

    Parameters
    ----------
    base_model_name:
        HuggingFace model ID or local path for the base model.
    adapter_paths:
        List of adapter paths (HF Hub repo IDs or local paths).
        Applied in order: adapter_paths[0] first, then [1], etc.
    output_dir:
        Directory to save the merged model.
    weights:
        Optional scaling weights for each adapter (default: 1.0 for all).
        Values < 1.0 reduce that adapter's influence.
    trust_remote_code:
        Trust remote code when loading models.
    torch_dtype:
        Data type: "float16", "bfloat16", or "float32".
    push_to_hub:
        Push merged model to HuggingFace Hub.
    hub_repo_id:
        Hub repository ID for pushing.
    hf_token:
        HuggingFace token for authentication.

    Returns
    -------
    Path to the merged model directory.
    """
    if not adapter_paths:
        raise ValueError("At least one adapter path is required")

    if weights and len(weights) != len(adapter_paths):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match "
            f"number of adapters ({len(adapter_paths)})"
        )

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    logger.info("Loading base model: %s", base_model_name)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=trust_remote_code,
    )

    # Apply adapters sequentially
    for i, adapter_path in enumerate(adapter_paths):
        weight = weights[i] if weights else 1.0
        logger.info(
            "Merging adapter %d/%d: %s (weight=%.2f)",
            i + 1, len(adapter_paths), adapter_path, weight,
        )

        # Load adapter
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )

        # Scale adapter weights if weight != 1.0
        if weight != 1.0:
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.data *= weight

        # Merge and unload (bakes LoRA into base weights)
        model = model.merge_and_unload()
        logger.info("  ✅ Merged adapter %d", i + 1)

        # Update tokenizer from latest adapter (it may have new tokens)
        try:
            adapter_tokenizer = AutoTokenizer.from_pretrained(
                adapter_path, trust_remote_code=trust_remote_code,
            )
            if len(adapter_tokenizer) > len(tokenizer):
                tokenizer = adapter_tokenizer
                model.resize_token_embeddings(len(tokenizer))
                logger.info("  Updated tokenizer from adapter %d (%d tokens)", i + 1, len(tokenizer))
        except Exception:
            pass

    # Save merged model
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("✅ Merged model saved to %s (%s params)", output_dir, f"{total_params:,}")

    # Push to Hub
    if push_to_hub and hub_repo_id:
        try:
            from huggingface_hub import HfApi

            api = HfApi(token=hf_token)
            api.upload_folder(
                folder_path=output_dir,
                repo_id=hub_repo_id,
                commit_message=f"Merged {len(adapter_paths)} LoRA adapters",
            )
            logger.info("✅ Merged model pushed to %s", hub_repo_id)
        except Exception as e:
            logger.warning("⚠️  Push failed: %s", e)

    return output_dir
