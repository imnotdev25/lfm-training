# 9. Architecture Deep-Dive

> **Goal**: How lfm-trainer is built — every module explained, the execution flow, and how to extend it.

---

## Package Structure

```
src/lfm_trainer/
├── __init__.py       # Public API exports
├── config.py         # TrainingConfig dataclass (all knobs)
├── data.py           # Data loading, formatting, quality filters
├── callbacks.py      # Error-resilient training (OOM, Ctrl+C, safe_train)
├── trainer.py        # 8-step training orchestrator
├── benchmark.py      # 5 coding benchmarks
├── model_card.py     # Auto-generated HF model card + Hub upload
├── export.py         # GGUF + MLX quantization & publishing
└── cli.py            # argparse CLI + entry point (lfm-train)
```

---

## Module-by-Module Walkthrough

### 1. `config.py` — The Single Source of Truth

Everything in lfm-trainer is controlled by one dataclass:

```python
@dataclass
class TrainingConfig:
    model_name: str                    # Base model to fine-tune
    dataset_paths: list[str]           # Data sources (Hub, local files)
    use_lora: bool = True              # LoRA vs full fine-tuning
    lora_r: int = 16                   # LoRA rank
    learning_rate: float = 2e-4        # Optimizer LR
    run_benchmark: bool = False        # Post-training eval
    export_gguf: bool = False          # Quantized export
    push_to_hub: bool = True           # Upload to HF
    ...
```

**Design decision**: No scattered config files or environment variables. Every setting lives here, making it easy to reproduce a training run.

**Auto-resolution**: `__post_init__` automatically resolves secrets:
- `hf_token` ← CLI arg → env `HF_TOKEN` → Kaggle Secrets
- `wandb_api_key` ← CLI arg → env `WANDB_API_KEY` → Kaggle Secrets

### 2. `data.py` — Data Pipeline

The data module handles the entire journey from raw files to tokenized tensors:

```
load_datasets()           # Entry point
├── _load_single()        # Load one source (Hub, CSV, JSON, Parquet)
├── _detect_format()      # Auto-detect: Alpaca, ShareGPT, tool-calling, text
├── _apply_formatters()   # Normalize everything to conversation format
├── clean_dataset()       # Quality filters (dedup, length, empty removal)
└── train_test_split()    # Eval split if requested
```

**Key design**: Auto-format detection. Users don't need to specify whether their data is Alpaca or ShareGPT — the module inspects column names and content to figure it out.

**Supported formats**:
- `instruction`, `input`, `output` → Alpaca
- `conversations` or `messages` → ShareGPT/conversational
- `text` → Plain text
- Any with `tool_calls` → Tool-calling

### 3. `callbacks.py` — Error Resilience

Training on cloud GPUs can fail. This module provides `safe_train()`:

```python
def safe_train(trainer, model, tokenizer, repo_id, token, ...):
    """Train with automatic recovery from:
    - OOM (Out of Memory) errors
    - SIGTERM signals (Kaggle timeout)
    - Keyboard interrupts (Ctrl+C)
    - Any unexpected exception
    """
```

**On any error**:
1. Save the current adapter/model to disk
2. Push to HuggingFace Hub (if configured)
3. Log what happened
4. Raise the original exception

This ensures you never lose training progress.

### 4. `trainer.py` — The Orchestrator

The main `run_training()` function executes an 8-step pipeline:

```
Step 1: Load base model
  └→ AutoModelForCausalLM.from_pretrained()
  └→ fp16 or bf16 precision
  └→ Auto device mapping (multi-GPU support)

Step 2: Configure LoRA or full fine-tuning
  └→ if use_lora: apply LoRA adapters (PeftModel)
  └→ if not use_lora: enable gradient checkpointing
  └→ if resume_from: load existing adapter from Hub or local

Step 3: Load and prepare dataset
  └→ data.load_datasets() with quality filters
  └→ Optional eval split

Step 4: Configure SFTTrainer
  └→ Set up SFTConfig (learning rate, batch size, etc.)
  └→ Configure W&B / TensorBoard reporting

Step 5: Train
  └→ safe_train() wraps trainer.train()
  └→ Saves checkpoints, pushes to Hub

Step 6: Post-training benchmarks
  └→ Optional: run HumanEval, MBPP, etc.
  └→ Optional: before/after comparison

Step 7: Model card generation
  └→ Generate README.md with config + results
  └→ Auto-upload to Hub

Step 8: Export
  └→ Merge LoRA (if applicable)
  └→ Quantize to GGUF and/or MLX
  └→ Push quantized versions to Hub
```

### 5. `benchmark.py` — Coding Evaluation

Five benchmark runners, all following the same pattern:

```python
# Common interface
def _run_benchmark(model, tokenizer, n_samples, max_problems) -> BenchmarkResult:
    for problem in dataset:
        completion = _generate_completion(model, tokenizer, problem.prompt)
        passed = _exec_safe(problem.prompt + completion + problem.tests)
        results.append(passed)
    return BenchmarkResult(pass_at_1=sum(results)/len(results))
```

**Registry pattern**: Benchmarks are registered in a dict for easy extension:

```python
_RUNNERS = {
    "humaneval": _run_humaneval,
    "mbpp": _run_mbpp,
    "multiple": _run_multiple,
    "bigcodebench": _run_bigcodebench,
    "evalplus": _run_evalplus,
}
```

To add a new benchmark, just write the runner function and add it to `_RUNNERS`.

### 6. `model_card.py` — Documentation Generator

Generates a HuggingFace-compatible README.md:

```markdown
---
base_model: liquid/LFM2.5-1.2B-Base
tags: [lfm-trainer, lora, peft, code]
---
# your-username/lfm-code

## Training Details
| Parameter | Value |
|-----------|-------|
| Base model | liquid/LFM2.5-1.2B-Base |
| LoRA rank | 16 |
...

## Benchmark Results
### HumanEval
| Metric | Before | After | Delta |
...
```

**Auto-upload**: After generation, uses `HfApi.upload_file()` to push to the Hub repo.

### 7. `export.py` — Quantization Pipeline

Handles GGUF and MLX export with shared versioning:

```python
def run_exports(model_dir, repo_id_base, version_tag, token, ...):
    if export_gguf:
        for quant in ["Q4_K_M", "Q6_K", "Q8_0"]:
            quantize(model_dir, quant)
            push_to_hub(f"{repo_id_base}-GGUF-{quant}")
    if export_mlx:
        for bits in [4, 6, 8]:
            mlx_convert(model_dir, bits)
            push_to_hub(f"{repo_id_base}-MLX-{bits}bit")
```

All variants share the same version tag (e.g., `v20260302-163200`).

### 8. `cli.py` — Command-Line Interface

Maps CLI arguments to `TrainingConfig`:

```bash
lfm-train --dataset X --hub-repo Y --benchmark --full-finetune
```

```python
# Internally:
args = parser.parse_args()
cfg = TrainingConfig(
    dataset_paths=args.dataset,
    hub_repo_id=args.hub_repo,
    run_benchmark=args.benchmark,
    use_lora=not args.full_finetune,
)
run_training(cfg)
```

---

## Execution Flow Diagram

```
User calls run_training(cfg)
         │
         ▼
┌─────────────────┐
│ 1. Load Model   │ → AutoModelForCausalLM + AutoTokenizer
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│ 2. LoRA/Full FT │ ──→ │ PeftModel (LoRA)  │ or gradient checkpointing
└────────┬────────┘     └──────────────────┘
         │
         ▼
┌─────────────────┐
│ 3. Load Data    │ → load_datasets() → clean → split
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. SFTConfig    │ → learning rate, batch size, reporting
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Train        │ → safe_train(trainer) → checkpoints → Hub
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ 6.Bench│ │ 7. Card  │ → README.md + benchmark_results.json
└────┬───┘ └──────────┘
     │
     ▼
┌─────────────────┐
│ 8. Export       │ → Merge LoRA → GGUF/MLX → Hub
└─────────────────┘
```

---

## Key Dependencies

| Library | What it does | Why we use it |
|---------|-------------|---------------|
| `torch` | Tensor operations, GPU compute | Foundation of all deep learning |
| `transformers` | Model loading, tokenization | HuggingFace model hub |
| `peft` | LoRA adapters | Parameter-efficient fine-tuning |
| `trl` | SFTTrainer | Supervised fine-tuning with chat templates |
| `datasets` | Data loading, processing | HuggingFace data hub |
| `accelerate` | Multi-GPU, mixed precision | Distributed training |
| `bitsandbytes` | 4/8-bit loading | Memory-efficient model loading |
| `huggingface-hub` | Upload/download models | Model publishing |

Optional:
| `wandb` | Experiment tracking | Training metrics dashboard |
| `tensorboard` | Metrics visualization | Alternative to W&B |
| `evalplus` | HumanEval+ benchmark | Extended test cases |
| `mlx-lm` | Apple Silicon export | Mac-optimized inference |

---

## How to Extend lfm-trainer

### Add a new benchmark

```python
# In benchmark.py:

def _run_my_benchmark(model, tokenizer, n_samples, max_problems):
    # Load your dataset
    ds = load_dataset("my-org/my-benchmark", split="test")
    # Generate and evaluate
    ...
    return BenchmarkResult(benchmark="MyBench", pass_at_1=score)

# Register it:
_RUNNERS["mybench"] = _run_my_benchmark
AVAILABLE_BENCHMARKS.append("mybench")
```

### Add a new data format

```python
# In data.py, add detection logic in _detect_format():

def _detect_format(dataset):
    columns = dataset.column_names
    if "my_special_column" in columns:
        return "my_format"
    ...

# And a formatter:
def _format_my_format(example):
    return {"text": format_as_conversation(example)}
```

### Add a new export format

```python
# In export.py:

def _export_my_format(model_dir, output_dir, ...):
    # Convert and save
    ...

# Wire it into run_exports()
```

---

## Summary

lfm-trainer is designed around three principles:

1. **Single config**: Everything controlled by `TrainingConfig`
2. **Modular pipeline**: Each step is independent and testable
3. **Error resilience**: Training never loses progress

The 8-step pipeline handles the entire lifecycle:
**Load → Configure → Data → Train → Benchmark → Document → Export → Publish**
