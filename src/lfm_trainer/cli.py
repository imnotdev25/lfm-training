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

    # ── LoRA / Full fine-tuning ─────────────────────────────────────────
    p.add_argument(
        "--full-finetune",
        action="store_true",
        help="Train all model parameters (no LoRA). Needs more VRAM but gives best results.",
    )
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
    p.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help=(
            "Which benchmarks to run (default: humaneval mbpp). "
            "Options: humaneval, mbpp, multiple, bigcodebench, evalplus, all."
        ),
    )

    # ── Logging / reporting ───────────────────────────────────────────
    p.add_argument(
        "--report-to",
        default="none",
        choices=["none", "wandb", "tensorboard"],
        help="Where to log training metrics (default: none).",
    )
    p.add_argument(
        "--wandb-key",
        default=None,
        help="W&B API key (auto-detected from WANDB_API_KEY env or Kaggle Secrets).",
    )
    p.add_argument(
        "--wandb-project",
        default="lfm-trainer",
        help="W&B project name (default: lfm-trainer).",
    )
    p.add_argument(
        "--no-model-card",
        action="store_true",
        help="Skip auto-generating a HuggingFace model card.",
    )

    # ── Alignment (DPO / PPO / GRPO) ────────────────────────────────────
    p.add_argument(
        "--alignment-method",
        choices=["dpo", "ppo", "grpo"],
        default="dpo",
        help="Alignment method: dpo, ppo, or grpo (default: dpo).",
    )
    p.add_argument(
        "--alignment-dataset",
        default=None,
        help="HF dataset for alignment stage (DPO: prompt/chosen/rejected; PPO/GRPO: prompts).",
    )
    p.add_argument(
        "--dpo-beta",
        type=float,
        default=0.1,
        help="DPO β parameter — higher is more conservative (default: 0.1).",
    )
    p.add_argument(
        "--reward-model",
        default=None,
        help="HF reward model for PPO (e.g., OpenAssistant/reward-model-deberta-v3-large).",
    )
    p.add_argument(
        "--grpo-generations",
        type=int,
        default=4,
        help="Number of completions per prompt for GRPO (default: 4).",
    )

    # ── Continued Pre-Training (CPT) ──────────────────────────────────
    p.add_argument(
        "--cpt-sources",
        nargs="+",
        default=None,
        metavar="PATH",
        help="Raw text sources for CPT: file paths, directories, or HF dataset IDs.",
    )
    p.add_argument(
        "--cpt-chunk-size",
        type=int,
        default=2048,
        help="Characters per chunk for CPT (default: 2048).",
    )

    # ── Reasoning ──────────────────────────────────────────────────────
    p.add_argument(
        "--enable-reasoning",
        action="store_true",
        help="Enable <think> reasoning tags in training data.",
    )
    p.add_argument(
        "--reasoning-dataset",
        default=None,
        help="HF dataset for reasoning training (e.g., LLM360/TxT360-3efforts).",
    )
    p.add_argument(
        "--reasoning-max-samples",
        type=int,
        default=100_000,
        help="Max samples from reasoning dataset (default: 100000).",
    )

    # ── Structured Output ────────────────────────────────────────────
    p.add_argument(
        "--structured-output",
        action="store_true",
        help="Mix in JSON schema training data for structured output.",
    )

    # ── Auto HP Search ────────────────────────────────────────────────
    p.add_argument(
        "--auto-hp-search",
        action="store_true",
        help="Run auto hyperparameter search before training.",
    )
    p.add_argument(
        "--hp-trial-steps",
        type=int,
        default=50,
        help="Steps per HP search trial (default: 50).",
    )

    # ── DeepSpeed ────────────────────────────────────────────────────
    p.add_argument(
        "--deepspeed",
        default=None,
        metavar="STAGE",
        help="DeepSpeed config: 'zero2', 'zero3', or path to custom JSON.",
    )

    # ── Model Distillation ───────────────────────────────────────────
    p.add_argument(
        "--distill-teacher",
        default=None,
        help="HF model ID of the teacher model for knowledge distillation.",
    )
    p.add_argument(
        "--distill-temperature",
        type=float,
        default=2.0,
        help="Distillation softmax temperature (default: 2.0).",
    )
    p.add_argument(
        "--distill-alpha",
        type=float,
        default=0.5,
        help="Blend factor: 0=CE only, 1=KL only (default: 0.5).",
    )

    # ── LoRA Merge ────────────────────────────────────────────────────
    p.add_argument(
        "--merge-adapters",
        nargs="+",
        default=None,
        metavar="ADAPTER",
        help="Merge multiple LoRA adapters into a single model (skip training).",
    )
    p.add_argument(
        "--merge-output",
        default="./lfm-merged",
        help="Output directory for merged model (default: ./lfm-merged).",
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
        use_lora=not args.full_finetune,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        fp16=not args.bf16,
        bf16=args.bf16,
        report_to=args.report_to,
        wandb_api_key=args.wandb_key,
        wandb_project=args.wandb_project,
        export_gguf=args.export_gguf,
        export_mlx=args.export_mlx,
        export_output_dir=args.export_dir,
        push_to_hub=not args.no_push,
        run_benchmark=args.benchmark,
        benchmark_before_after=args.benchmark_compare,
        benchmark_max_problems=args.benchmark_max,
        benchmark_names=(
            ["humaneval", "mbpp", "multiple", "bigcodebench", "evalplus"]
            if args.benchmarks and "all" in args.benchmarks
            else args.benchmarks
        ),
        generate_model_card=not args.no_model_card,
        alignment_method=args.alignment_method,
        alignment_dataset=args.alignment_dataset,
        dpo_beta=args.dpo_beta,
        reward_model=args.reward_model,
        grpo_num_generations=args.grpo_generations,
        cpt_sources=args.cpt_sources,
        cpt_chunk_size=args.cpt_chunk_size,
        enable_reasoning=args.enable_reasoning,
        reasoning_dataset=args.reasoning_dataset,
        reasoning_max_samples=args.reasoning_max_samples,
        auto_hp_search=args.auto_hp_search,
        hp_search_trials_steps=args.hp_trial_steps,
        deepspeed=args.deepspeed,
        distill_teacher=args.distill_teacher,
        distill_temperature=args.distill_temperature,
        distill_alpha=args.distill_alpha,
        structured_output=args.structured_output,
        simulate_error=args.simulate_error,
    )

    # ── LoRA merge mode (skip training) ───────────────────────────────
    if args.merge_adapters:
        from lfm_trainer.merge import merge_adapters

        merge_adapters(
            base_model_name=cfg.model_name,
            adapter_paths=args.merge_adapters,
            output_dir=args.merge_output,
            push_to_hub=cfg.push_to_hub,
            hub_repo_id=cfg.hub_repo_id,
            hf_token=cfg.hf_token,
        )
        return

    run_training(cfg)


if __name__ == "__main__":
    main()
