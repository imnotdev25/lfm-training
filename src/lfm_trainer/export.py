"""
Post-training quantization and export to GGUF and MLX formats.

Produces 4-bit, 6-bit, and 8-bit quantized versions and pushes them
to HuggingFace Hub with the same version tag as the base model.
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from huggingface_hub import HfApi

from lfm_trainer.turboquant import calibrate_and_save_turboquant

logger = logging.getLogger(__name__)

# ── Quantization bit-widths ───────────────────────────────────────────────
QUANT_BITS = [4, 6, 8]

# GGUF quantization type mapping (llama.cpp naming)
GGUF_QUANT_MAP = {
    4: "Q4_K_M",
    6: "Q6_K",
    8: "Q8_0",
}


def _ensure_llama_cpp() -> Path:
    """Clone llama.cpp if not already present. Returns the repo path."""
    llama_dir = Path("/tmp/llama.cpp")
    if not llama_dir.exists():
        logger.info("Cloning llama.cpp for GGUF conversion…")
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", str(llama_dir)],
            check=True,
        )
        # Install Python deps for the convert script
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(llama_dir / "requirements.txt")],
            check=True,
            capture_output=True,
        )
    return llama_dir


# ── GGUF Export ───────────────────────────────────────────────────────────

def export_gguf(
    model_dir: str,
    output_base: str,
    repo_id_base: str,
    version_tag: str,
    token: Optional[str] = None,
) -> list[str]:
    """Convert a HuggingFace model directory to GGUF at multiple quantizations.

    Steps per bit-width:
      1. Convert safetensors → GGUF F16
      2. Quantize F16 → target quant type
      3. Push to Hub as ``{repo_id_base}-GGUF-{quant}``

    Returns a list of Hub repo IDs that were published.
    """
    llama_dir = _ensure_llama_cpp()
    convert_script = llama_dir / "convert_hf_to_gguf.py"

    # Step 1: Convert to F16 GGUF
    f16_path = Path(output_base) / "gguf-f16.gguf"
    f16_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Converting to GGUF F16: %s → %s", model_dir, f16_path)
    try:
        result = subprocess.run(
            [
                sys.executable, str(convert_script),
                model_dir,
                "--outfile", str(f16_path),
                "--outtype", "f16",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        logger.error("❌ GGUF F16 conversion failed (exit %d)", e.returncode)
        if e.stderr:
            logger.error("   stderr: %s", e.stderr[-500:])

        # Fallback: try using transformers' built-in GGUF support
        logger.info("Attempting fallback GGUF export via transformers…")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            _model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype="auto")
            _tokenizer = AutoTokenizer.from_pretrained(model_dir)
            _model.save_pretrained(str(f16_path.parent / "gguf-transformers"), safe_serialization=True)
            logger.warning(
                "⚠️  llama.cpp converter does not support this model architecture. "
                "Saved safetensors to %s instead. "
                "You can convert manually once llama.cpp adds support for this arch, or "
                "use the safetensors model directly.",
                f16_path.parent / "gguf-transformers",
            )
            return []  # Skip quantization steps
        except Exception as fallback_err:
            logger.error("❌ Fallback export also failed: %s", fallback_err)
            return []

    published_repos: list[str] = []
    api = HfApi(token=token)

    for bits in QUANT_BITS:
        quant_type = GGUF_QUANT_MAP[bits]
        quant_path = Path(output_base) / f"gguf-{quant_type}.gguf"
        repo_id = f"{repo_id_base}-GGUF"

        # Step 2: Quantize
        quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            # Try alternative path or use the Python quantize script
            quantize_bin = llama_dir / "llama-quantize"

        if quantize_bin.exists():
            logger.info("Quantizing GGUF: %s → %s (%s)", f16_path, quant_path, quant_type)
            subprocess.run(
                [str(quantize_bin), str(f16_path), str(quant_path), quant_type],
                check=True,
            )
        else:
            # Fallback: use convert script with direct quantization
            logger.info("Quantize binary not found, converting directly at %s", quant_type)
            subprocess.run(
                [
                    sys.executable, str(convert_script),
                    model_dir,
                    "--outfile", str(quant_path),
                    "--outtype", quant_type.lower(),
                ],
                check=True,
            )

        # Step 3: Push to Hub
        logger.info("Pushing GGUF %s to %s (tag: %s)", quant_type, repo_id, version_tag)
        try:
            api.create_repo(repo_id, exist_ok=True, token=token)
            api.upload_file(
                path_or_fileobj=str(quant_path),
                path_in_repo=f"{quant_type}/{quant_path.name}",
                repo_id=repo_id,
                token=token,
                commit_message=f"Add GGUF {quant_type} — {version_tag}",
            )
            api.create_tag(repo_id=repo_id, tag=version_tag, tag_message=f"GGUF {quant_type}")
            published_repos.append(repo_id)
            logger.info("✅ Published GGUF %s → %s", quant_type, repo_id)
        except Exception as e:
            logger.error("❌ Failed to push GGUF %s: %s", quant_type, e)

    return published_repos


# ── MLX Export ────────────────────────────────────────────────────────────

def export_mlx(
    model_dir: str,
    output_base: str,
    repo_id_base: str,
    version_tag: str,
    token: Optional[str] = None,
) -> list[str]:
    """Convert a HuggingFace model to MLX format at 4/6/8-bit via ``mlx-lm``.

    Uses the ``mlx_lm.convert`` Python API with ``--upload-repo`` to push
    each quantized variant to HuggingFace.

    Returns a list of Hub repo IDs that were published.
    """
    try:
        from mlx_lm import convert as mlx_convert
    except ImportError:
        logger.warning(
            "mlx-lm is not installed. Skipping MLX export. "
            "Install with: pip install mlx-lm"
        )
        return []

    api = HfApi(token=token)
    published_repos: list[str] = []

    for bits in QUANT_BITS:
        suffix = f"MLX-{bits}bit"
        upload_repo = f"{repo_id_base}-{suffix}"
        mlx_output = Path(output_base) / f"mlx-{bits}bit"

        logger.info("Converting to MLX %d-bit: %s → %s", bits, model_dir, upload_repo)

        try:
            mlx_convert(
                model_dir,
                quantize=True,
                q_bits=bits,
                mlx_path=str(mlx_output),
                upload_repo=upload_repo,
            )
            # Tag the repo with the shared version
            api.create_tag(
                repo_id=upload_repo,
                tag=version_tag,
                tag_message=f"MLX {bits}-bit quantization",
            )
            published_repos.append(upload_repo)
            logger.info("✅ Published MLX %d-bit → %s", bits, upload_repo)
        except Exception as e:
            logger.error("❌ Failed MLX %d-bit export: %s", bits, e)

    return published_repos


def export_turboquant(
    model_dir: str,
    output_base: str,
    repo_id_base: str,
    version_tag: str,
    token: Optional[str] = None,
    dtype: str = "turboquant25",
    max_prompts: int = 128,
    max_seq_len: int = 512,
    calibration_data: Optional[list[str]] = None,
) -> list[str]:
    """Generate TurboQuant metadata for KV cache quantization.

    Returns a list of Hub repo IDs that were published (if any).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    api = HfApi(token=token)
    suffix = "TurboQuant"
    upload_repo = f"{repo_id_base}-{suffix}"
    tq_output_dir = Path(output_base) / "turboquant"
    tq_output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = tq_output_dir / "turboquant_metadata.json"

    logger.info("Generating TurboQuant metadata: %s", model_dir)

    try:
        # Load model and tokenizer for calibration
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        calibrate_and_save_turboquant(
            model=model,
            tokenizer=tokenizer,
            output_path=str(metadata_path),
            calibration_data=calibration_data,
            kv_cache_dtype=dtype,
            max_prompts=max_prompts,
            max_seq_len=max_seq_len,
        )

        # Upload the metadata to the Hub
        logger.info("Uploading TurboQuant metadata to %s", upload_repo)
        api.create_repo(repo_id=upload_repo, exist_ok=True)
        
        # Upload model config files and the metadata
        api.upload_folder(
            folder_path=model_dir,
            repo_id=upload_repo,
            ignore_patterns=["*.safetensors", "*.bin", "*.pth"],
        )
        api.upload_file(
            path_or_fileobj=str(metadata_path),
            path_in_repo="turboquant_metadata.json",
            repo_id=upload_repo,
        )

        # Tag the repo
        api.create_tag(
            repo_id=upload_repo,
            tag=version_tag,
            tag_message="TurboQuant metadata generation",
        )
        
        logger.info("✅ Published TurboQuant metadata → %s", upload_repo)
        return [upload_repo]

    except Exception as e:
        logger.error("❌ Failed TurboQuant export: %s", e)
        return []


# ── Unified export runner ─────────────────────────────────────────────────

def run_exports(
    model_dir: str,
    repo_id_base: str,
    version_tag: str,
    token: Optional[str] = None,
    output_base: str = "./lfm-exports",
    enable_gguf: bool = True,
    enable_mlx: bool = True,
    enable_turboquant: bool = False,
    turboquant_dtype: str = "turboquant25",
    turboquant_max_prompts: int = 128,
    turboquant_max_seq_len: int = 512,
    calibration_data: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """Run all post-training exports with a shared version tag.

    Parameters
    ----------
    model_dir:
        Path to the trained HuggingFace model directory (with safetensors).
    repo_id_base:
        Base Hub repo ID. Quant variants are named ``{base}-GGUF``,
        ``{base}-MLX-4bit``, etc.
    version_tag:
        Shared version tag (e.g. ``v20260302-160000``) applied to all repos.
    token:
        HuggingFace API token.
    output_base:
        Local directory for intermediate export files.
    enable_gguf:
        Whether to export GGUF variants.
    enable_mlx:
        Whether to export MLX variants.
    enable_turboquant:
        Whether to generate TurboQuant metadata.
    turboquant_dtype:
        TurboQuant KV cache dtype ("turboquant25" or "turboquant35").
    turboquant_max_prompts:
        Max samples for calibration.
    turboquant_max_seq_len:
        Max sequence length for calibration.
    calibration_data:
        Optional list of strings to use for calibration.

    Returns
    -------
    Dict mapping format name to list of published repo IDs.
    """
    results: dict[str, list[str]] = {}

    # First, tag the base model repo with the same version
    try:
        api = HfApi(token=token)
        api.create_tag(
            repo_id=repo_id_base,
            tag=version_tag,
            tag_message="Base model (full precision)",
        )
        logger.info("🏷️  Tagged base repo %s with %s", repo_id_base, version_tag)
    except Exception as e:
        logger.warning("Could not tag base repo: %s", e)

    if enable_gguf:
        logger.info("═══ Starting GGUF export ═══")
        results["gguf"] = export_gguf(model_dir, output_base, repo_id_base, version_tag, token)

    if enable_mlx:
        logger.info("═══ Starting MLX export ═══")
        results["mlx"] = export_mlx(model_dir, output_base, repo_id_base, version_tag, token)

    if enable_turboquant:
        logger.info("═══ Starting TurboQuant export ═══")
        results["turboquant"] = export_turboquant(
            model_dir,
            output_base,
            repo_id_base,
            version_tag,
            token,
            dtype=turboquant_dtype,
            max_prompts=turboquant_max_prompts,
            max_seq_len=turboquant_max_seq_len,
            calibration_data=calibration_data,
        )

    # Summary
    total = sum(len(v) for v in results.values())
    logger.info(
        "🎉 Export complete — %d variants published under version %s",
        total, version_tag,
    )
    return results
