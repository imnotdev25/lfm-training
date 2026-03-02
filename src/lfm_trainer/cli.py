"""
CLI entry-point for ``lfm-train``.

Usage examples::

    # Minimal — uses Kaggle secrets for HF token
    lfm-train --dataset my_data.csv

    # Multiple datasets
    lfm-train --dataset data1.csv --dataset data2.parquet --dataset iamtarun/python_code_instructions_18k_alpaca

    # Full control
    lfm-train \
        --model liquid/LFM2.5-1.2B-Base \
        --dataset code_data.jsonl \
        --hf-token hf_xxxx \
        --hub-repo my-org/lfm-code \
        --epochs 5 \
        --batch-size 4 \
        --lr 1e-4 \
        --max-seq-length 4096 \
        --output-dir ./my-checkpoints

    # Test auto-publish on simulated error
    lfm-train --dataset dummy.csv --simulate-error
"""

from __future__ import annotations

import argparse
import logging
import sys

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lfm-train",
        description="Fine-tune Liquid LFM 2.5 1.2B for coding tasks with auto-publish on error.",
    )

    # ── Dataset ────────────────────────────────────────────────────────
    p.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        required=True,
        help=(
            "Dataset source — local path (CSV/Parquet/JSONL) or a HuggingFace Hub ID. "
            "Can be specified multiple times to merge datasets."
        ),
    )
    p.add_argument(
        "--text-column",
        default="text",
        help="Name of the text column if using a single-column dataset (default: text).",
    )
    p.add_argument(
        "--tool-calling-only",
        action="store_true",
        help=(
            "Filter dataset to keep only samples containing tool calls. "
            "Useful for datasets like jdaddyalbs/playwright-mcp-toolcalling."
        ),
    )

    # ── Model ──────────────────────────────────────────────────────────
    p.add_argument(
        "--model",
        default="liquid/LFM2.5-1.2B-Base",
        help="HuggingFace model ID (default: liquid/LFM2.5-1.2B-Base).",
    )
    p.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Path or Hub ID of a previously trained LoRA adapter to continue "
            "training on a new dataset. Skips fresh LoRA init and loads the "
            "existing adapter instead."
        ),
    )

    # ── Hub / auth ─────────────────────────────────────────────────────
    p.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token. Also read from HF_TOKEN env var or Kaggle Secrets.",
    )
    p.add_argument(
        "--hub-repo",
        default=None,
        help="Hub repo ID to push to (default: auto-generated).",
    )
    p.add_argument(
        "--no-push",
        action="store_true",
        help=(
            "Save model locally only, do not push to HuggingFace Hub on success. "
            "Useful for intermediate continual-training rounds."
        ),
    )

    # ── Training hyper-params ──────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--max-seq-length", type=int, default=2048)

    # ── LoRA ───────────────────────────────────────────────────────────
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # ── Output ─────────────────────────────────────────────────────────
    p.add_argument("--output-dir", default="./lfm-checkpoints")
    p.add_argument("--save-steps", type=int, default=100)

    # ── Precision ──────────────────────────────────────────────────────
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 instead of float16.")

    # ── Post-training export ───────────────────────────────────────────
    p.add_argument(
        "--export-gguf",
        action="store_true",
        help="After training, export GGUF quantized versions (Q4_K_M, Q6_K, Q8_0).",
    )
    p.add_argument(
        "--export-mlx",
        action="store_true",
        help="After training, export MLX quantized versions (4-bit, 6-bit, 8-bit).",
    )
    p.add_argument(
        "--export-dir",
        default="./lfm-exports",
        help="Directory for intermediate export files (default: ./lfm-exports).",
    )

    # ── Data quality ───────────────────────────────────────────────────
    p.add_argument(
        "--quality-filter",
        action="store_true",
        help="Remove empty rows, length outliers, and duplicates from the dataset.",
    )
    p.add_argument(
        "--eval-split",
        type=float,
        default=0.0,
        help="Hold out a fraction for evaluation (e.g. 0.1 = 10%%). Reports eval loss during training.",
    )

    # ── Benchmarking ──────────────────────────────────────────────────
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run HumanEval + MBPP benchmarks after training.",
    )
    p.add_argument(
        "--benchmark-compare",
        action="store_true",
        help="Also benchmark the base model before training for a before/after comparison.",
    )
    p.add_argument(
        "--benchmark-max",
        type=int,
        default=None,
        help="Cap the number of benchmark problems (for quick testing).",
    )

    # ── Logging / reporting ───────────────────────────────────────────
    p.add_argument(
        "--report-to",
        default="none",
        choices=["none", "wandb", "tensorboard"],
        help="Where to log training metrics (default: none).",
    )
    p.add_argument(
        "--no-model-card",
        action="store_true",
        help="Skip auto-generating a HuggingFace model card.",
    )

    # ── Debug ──────────────────────────────────────────────────────────
    p.add_argument(
        "--simulate-error",
        action="store_true",
        help="Simulate a training error after 5 steps to test auto-publish.",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    """Entry point invoked by the ``lfm-train`` console script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    cfg = TrainingConfig(
        model_name=args.model,
        resume_from_model=args.resume_from,
        dataset_paths=args.datasets,
        dataset_text_column=args.text_column,
        tool_calling_only=args.tool_calling_only,
        quality_filter=args.quality_filter,
        eval_split=args.eval_split,
        hf_token=args.hf_token,
        hub_repo_id=args.hub_repo,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        fp16=not args.bf16,
        bf16=args.bf16,
        report_to=args.report_to,
        export_gguf=args.export_gguf,
        export_mlx=args.export_mlx,
        export_output_dir=args.export_dir,
        push_to_hub=not args.no_push,
        run_benchmark=args.benchmark,
        benchmark_before_after=args.benchmark_compare,
        benchmark_max_problems=args.benchmark_max,
        generate_model_card=not args.no_model_card,
        simulate_error=args.simulate_error,
    )

    run_training(cfg)


if __name__ == "__main__":
    main()
