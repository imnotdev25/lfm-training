# LFM Trainer

Fine-tune **Liquid LFM 2.5 1.2B** for coding tasks on Kaggle multi-GPU — with automatic checkpoint publishing to Hugging Face on errors, and post-training GGUF + MLX quantization.

## Features

- 🚀 **Multi-GPU** training via HuggingFace Accelerate / DDP (Kaggle P100 / 2×T4)
- 🧠 **LoRA / PEFT** for memory-efficient fine-tuning
- 📦 **Structured dataset loading** — CSV, Parquet, JSONL, HuggingFace Hub IDs, or direct `pd.DataFrame` objects
- 🔍 **Auto-format detection** — Alpaca, prompt/response, conversational/chat (DataClaw), single text column
- �️ **Tool calling support** — LFM 2.5 native `<|tool_call_start|>` / `<|tool_call_end|>` tokens; handles OpenAI and DataClaw tool call formats
- �🛡️ **Error-resilient training** — auto-publishes versioned checkpoints on OOM, SIGTERM (Kaggle timeout), KeyboardInterrupt, or any exception
- 🔑 **Flexible HF auth** — CLI arg, `HF_TOKEN` env var, or Kaggle Secrets
- 📐 **GGUF export** — Q4_K_M, Q6_K, Q8_0 via llama.cpp
- 🍎 **MLX export** — 4-bit, 6-bit, 8-bit via [mlx-lm](https://github.com/ml-explore/mlx-lm)
- 🏷️ **Shared versioning** — base model + all quantized variants tagged with the same version
- 🔓 **Full fine-tuning** — train all parameters (no LoRA) for maximum quality
- 📊 **Auto-benchmarking** — HumanEval + MBPP + MultiPL-E + BigCodeBench + EvalPlus
- 🧹 **Data quality filters** — auto-remove duplicates, empty rows, and length outliers
- 📈 **Eval split** — hold out a % for validation loss tracking during training
- 📝 **Auto model card** — generates a HuggingFace README.md with config, benchmarks, and hardware
- 📉 **W&B / TensorBoard** — optional training metric logging
- 🎯 **DPO / PPO / GRPO alignment** — preference tuning after SFT via DPO, classic RLHF (PPO), or DeepSeek-style GRPO
- 📚 **Continued Pre-Training (CPT)** — train on raw text (books, PDFs, code) to inject domain knowledge
- 🔗 **LoRA adapter merging** — combine multiple adapters into a single model with weighted blending
- 🔍 **Auto HP search** — try multiple learning rates and pick the best based on eval loss

## Installation

```bash
pip install lfm-trainer
```

## Quick Start (Kaggle Notebook)

```python
# Cell 1: Install
!pip install lfm-trainer

# Cell 2: Train on a coding dataset with GGUF export
!lfm-train \
    --dataset peteromallet/dataclaw-peteromallet \
    --hub-repo your-username/lfm-code \
    --export-gguf

# Train on multiple datasets
!lfm-train \
    --dataset code_data.csv \
    --dataset more_code.parquet \
    --dataset sahil2801/CodeAlpaca-20k \
    --hub-repo your-username/lfm-code \
    --epochs 3 \
    --batch-size 2 \
    --export-gguf
```

The HF token is automatically picked up from Kaggle Secrets (key: `HF_TOKEN`).

## Python API

```python
import pandas as pd
from lfm_trainer import run_training, load_datasets
from lfm_trainer.config import TrainingConfig

# Load multiple sources including DataFrames
df = pd.read_csv("my_code_data.csv")
dataset = load_datasets([
    df,                                    # Direct DataFrame
    "code_data.parquet",                   # Local file
    "peteromallet/dataclaw-peteromallet",   # HuggingFace Hub (conversational)
])

# Or use the full pipeline
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["peteromallet/dataclaw-peteromallet"],
    hub_repo_id="your-username/lfm-code",
    quality_filter=True,
    eval_split=0.1,
    run_benchmark=True,
    benchmark_before_after=True,
    export_gguf=True,
)
run_training(cfg)
```

## CLI Reference

```
lfm-train --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | *(required)* | Dataset path or Hub ID (repeatable) |
| `--model` | `liquid/LFM2.5-1.2B-Base` | Model to fine-tune |
| `--resume-from` | — | Path or Hub ID of a prior adapter for continual training |
| `--tool-calling-only` | off | Keep only samples with tool calls |
| `--quality-filter` | off | Remove empty rows, dupes, length outliers |
| `--eval-split` | 0.0 | Hold out a fraction for eval (e.g. 0.1 = 10%) |
| `--hf-token` | *auto-detect* | HuggingFace token |
| `--hub-repo` | auto | Hub repo to push to |
| `--no-push` | off | Save locally only, skip Hub push |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 2 | Per-device batch size |
| `--lr` | 2e-4 | Learning rate |
| `--max-seq-length` | 2048 | Max sequence length |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--bf16` | off | Use bfloat16 |
| `--full-finetune` | off | Train all params (no LoRA), needs more VRAM |
| `--report-to` | `none` | `none`, `wandb`, or `tensorboard` |
| `--benchmark` | off | Run HumanEval + MBPP after training |
| `--benchmark-compare` | off | Also benchmark base model for delta |
| `--benchmark-max` | all | Cap problems for quick testing |
| `--no-model-card` | off | Skip auto model card generation |
| `--export-gguf` | off | Export GGUF (Q4_K_M, Q6_K, Q8_0) |
| `--export-mlx` | off | Export MLX (4/6/8-bit) |
| `--export-dir` | `./lfm-exports` | Export scratch directory |
| `--alignment-method` | `dpo` | Alignment method: dpo, ppo, or grpo |
| `--alignment-dataset` | *none* | HF dataset for alignment (DPO: chosen/rejected; PPO/GRPO: prompts) |
| `--dpo-beta` | 0.1 | DPO β — higher = more conservative |
| `--reward-model` | *none* | HF reward model for PPO |
| `--grpo-generations` | 4 | Completions per prompt for GRPO |
| `--cpt-sources` | *none* | Raw text sources for CPT (files, dirs, HF datasets) |
| `--cpt-chunk-size` | 2048 | Characters per chunk for CPT |
| `--auto-hp-search` | off | Run auto hyperparameter search before training |
| `--hp-trial-steps` | 50 | Steps per HP search trial |
| `--merge-adapters` | *none* | Merge multiple LoRA adapters (skip training) |
| `--merge-output` | `./lfm-merged` | Output dir for merged model |
| `--simulate-error` | off | Test auto-publish mechanism |

## Supported Dataset Formats

The data loader auto-detects column layouts:

| Format | Detected By | Example |
|--------|------------|---------|
| **Alpaca** | `instruction` + `output` columns | `iamtarun/python_code_instructions_18k_alpaca` |
| **Prompt/Response** | `prompt` + `response` columns | Generic Q&A datasets |
| **Conversational** | `messages` column (with tool calls) | `peteromallet/dataclaw-peteromallet` |
| **Text** | Single `text` column | `jdaddyalbs/playwright-mcp-toolcalling` |
| **DataFrame** | Direct `pd.DataFrame` objects | In-memory data |

## Tool-Calling Training

Train exclusively on tool-calling examples using `--tool-calling-only`:

```bash
lfm-train \
    --dataset jdaddyalbs/playwright-mcp-toolcalling \
    --tool-calling-only \
    --hub-repo your-username/lfm-tools \
    --max-seq-length 4096
```

The filter keeps only samples containing tool call patterns (`<|tool_call_start|>`, `tool_calls`, `function_call`, etc.). Works with any dataset — pre-formatted or auto-formatted.

## Continual Training

Train iteratively across multiple datasets, saving locally between rounds:

```bash
# Round 1: Coding skills → save locally
lfm-train --dataset sahil2801/CodeAlpaca-20k --no-push

# Round 2: Add tool-calling on top → save locally
lfm-train \
    --resume-from ./lfm-checkpoints/final-adapter \
    --dataset jdaddyalbs/playwright-mcp-toolcalling \
    --tool-calling-only \
    --output-dir ./lfm-checkpoints-r2 \
    --no-push

# Round 3: Final round → push to Hub + export
lfm-train \
    --resume-from ./lfm-checkpoints-r2/final-adapter \
    --dataset peteromallet/dataclaw-peteromallet \
    --hub-repo your-username/lfm-final \
    --export-gguf
```

Each `--resume-from` loads the prior adapter and continues learning from where it left off.

## How Auto-Publish Works

Training is wrapped in an error handler inspired by Unsloth:

1. **SIGTERM** (Kaggle timeout) → saves + pushes immediately, then exits
2. **CUDA OOM** → clears cache, saves + pushes
3. **KeyboardInterrupt** → saves + pushes
4. **Any Exception** → saves + pushes, then re-raises

Each checkpoint is tagged with a UTC timestamp (e.g., `v20260302-153000`) so versions never collide.

## Post-Training Export

When `--export-gguf` or `--export-mlx` is enabled:

1. LoRA adapters are **merged** into the base model
2. **GGUF**: Converted via llama.cpp → Q4_K_M, Q6_K, Q8_0 → pushed to `{repo}-GGUF`
3. **MLX**: Converted via mlx-lm → 4-bit, 6-bit, 8-bit → pushed to `{repo}-MLX-{N}bit`
4. All repos (base + quants) share the **same version tag**

> **Note:** MLX export requires Apple Silicon. Use `--export-gguf` on Kaggle (Linux), and `--export-mlx` locally on Mac.

## Examples

See the [`examples/`](examples/) directory for ready-to-run scripts:

| Example | Description |
|---------|-------------|
| [`basic_training.py`](examples/basic_training.py) | Simple Alpaca coding fine-tune |
| [`tool_calling_training.py`](examples/tool_calling_training.py) | Tool-calling-only with playwright MCP |
| [`multi_dataset_training.py`](examples/multi_dataset_training.py) | Combining Hub + local + DataFrame sources |
| [`continual_training.py`](examples/continual_training.py) | Multi-round training with local saves |
| [`benchmark_training.py`](examples/benchmark_training.py) | Train + HumanEval/MBPP benchmark + auto-upload |
| [`full_benchmark_suite.py`](examples/full_benchmark_suite.py) | All 5 benchmarks with before/after comparison |
| [`benchmark_only.py`](examples/benchmark_only.py) | Evaluate any model without training |
| [`full_finetune.py`](examples/full_finetune.py) | Full parameter training (no LoRA) |
| [`wandb_training.py`](examples/wandb_training.py) | W&B logging with auto API key detection |
| [`export_only.py`](examples/export_only.py) | Standalone GGUF/MLX quantization |
| [`kaggle_notebook.py`](examples/kaggle_notebook.py) | Copy-paste Kaggle cells |
| [`cpt_raw_text.py`](examples/cpt_raw_text.py) | Train on books, PDFs, or raw text (CPT) |
| [`dpo_alignment.py`](examples/dpo_alignment.py) | DPO / PPO / GRPO alignment after SFT |
| [`merge_adapters.py`](examples/merge_adapters.py) | Merge multiple LoRA adapters (with weights) |
| [`auto_hp_search.py`](examples/auto_hp_search.py) | Auto learning rate search before training |

## 📚 Documentation

New to LLM fine-tuning? Start here — no prerequisites beyond basic Python:

| # | Guide | What you'll learn |
|---|-------|-------------------|
| 1 | [What is an LLM?](docs/01-what-is-an-llm.md) | Tokens, embeddings, attention, transformers |
| 2 | [How Training Works](docs/02-how-training-works.md) | Loss functions, backprop, gradient descent |
| 3 | [Fine-Tuning vs Scratch](docs/03-fine-tuning-explained.md) | Transfer learning, catastrophic forgetting |
| 4 | [LoRA Explained](docs/04-lora-explained.md) | Low-rank adapters, the math, pure Python impl |
| 5 | [Full Fine-Tuning](docs/05-full-fine-tuning.md) | Gradient checkpointing, memory management |
| 6 | [Data Preparation](docs/06-data-preparation.md) | Formats, tokenization, quality filters |
| 7 | [Evaluation & Benchmarks](docs/07-evaluation-and-benchmarks.md) | HumanEval, MBPP, pass@k metric |
| 8 | [Quantization & Export](docs/08-quantization-and-export.md) | GGUF, MLX, INT4/INT8 |
| 9 | [Architecture Deep-Dive](docs/09-architecture-deep-dive.md) | How lfm-trainer is built |
| 10 | [DPO, PPO, GRPO & Alignment](docs/10-dpo-and-alignment.md) | DPO/PPO/GRPO math, datasets, pipeline |
| 11 | [Continued Pre-Training (CPT)](docs/11-continued-pretraining.md) | Train on books, raw text, domain knowledge |

## License

MIT
