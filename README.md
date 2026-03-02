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
    export_gguf=True,
    export_mlx=True,   # Requires Apple Silicon
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
| `--hf-token` | *auto-detect* | HuggingFace token |
| `--hub-repo` | auto | Hub repo to push to |
| `--epochs` | 3 | Training epochs |
| `--batch-size` | 2 | Per-device batch size |
| `--lr` | 2e-4 | Learning rate |
| `--max-seq-length` | 2048 | Max sequence length |
| `--lora-r` | 16 | LoRA rank |
| `--lora-alpha` | 32 | LoRA alpha |
| `--bf16` | off | Use bfloat16 |
| `--export-gguf` | off | Export GGUF (Q4_K_M, Q6_K, Q8_0) |
| `--export-mlx` | off | Export MLX (4/6/8-bit) |
| `--export-dir` | `./lfm-exports` | Export scratch directory |
| `--simulate-error` | off | Test auto-publish mechanism |

## Supported Dataset Formats

The data loader auto-detects column layouts:

| Format | Detected By | Example |
|--------|------------|---------|
| **Alpaca** | `instruction` + `output` columns | `iamtarun/python_code_instructions_18k_alpaca` |
| **Prompt/Response** | `prompt` + `response` columns | Generic Q&A datasets |
| **Conversational** | `messages` column (with tool calls) | `peteromallet/dataclaw-peteromallet` |
| **Text** | Single `text` column | Pre-formatted datasets |
| **DataFrame** | Direct `pd.DataFrame` objects | In-memory data |

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

## License

MIT
