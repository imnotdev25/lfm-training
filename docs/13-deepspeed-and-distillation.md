# 13. DeepSpeed & Model Distillation

> **Goal**: Scale training to multiple GPUs with DeepSpeed ZeRO, and compress a large model into a smaller one through knowledge distillation.

---

## Part 1: DeepSpeed ZeRO

### What is ZeRO?

ZeRO (**Z**ero **R**edundancy **O**ptimizer) eliminates memory redundancy in data-parallel training. Instead of each GPU holding a complete copy of everything, ZeRO shards data across GPUs:

| Stage | What's sharded | Memory savings | Overhead |
|-------|---------------|----------------|----------|
| **ZeRO-2** | Optimizer states + gradients | ~8× | Low |
| **ZeRO-3** | + Model weights | ~16× | Medium |

### When to use what

```
Single GPU → No DeepSpeed needed
2× T4 + LoRA → ZeRO-2 ✓
2× A100 + Full fine-tune → ZeRO-2 or ZeRO-3
4+ GPUs + 7B model → ZeRO-3 ✓
```

### Usage

Built-in ZeRO configs are shipped with lfm-trainer — no need to write JSON manually:

```bash
# ZeRO-2: LoRA fine-tuning on 2×T4
lfm-train --dataset sahil2801/CodeAlpaca-20k \
    --deepspeed zero2 \
    --hub-repo your-username/lfm-ds

# ZeRO-3: Full fine-tuning on multi-GPU
lfm-train --dataset sahil2801/CodeAlpaca-20k \
    --deepspeed zero3 \
    --full-finetune \
    --hub-repo your-username/lfm-ds-full

# Custom DeepSpeed config
lfm-train --dataset sahil2801/CodeAlpaca-20k \
    --deepspeed /path/to/my_config.json
```

### Python API

```python
from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    deepspeed="zero2",     # or "zero3" or "/path/to/config.json"
    use_lora=True,
    bf16=True,
)
run_training(cfg)
```

### How ZeRO-2 works

```
┌─────────────────────────────────────────────┐
│               Standard Data Parallel         │
│                                              │
│  GPU 0: Full Model + Full Optimizer          │
│  GPU 1: Full Model + Full Optimizer          │
│  → 2× memory waste                           │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│               ZeRO Stage 2                   │
│                                              │
│  GPU 0: Full Model + Optimizer Shard A       │
│  GPU 1: Full Model + Optimizer Shard B       │
│  → Optimizer memory halved!                  │
│  → Gradients communicated via all-reduce     │
└─────────────────────────────────────────────┘
```

### How ZeRO-3 works

```
┌─────────────────────────────────────────────┐
│               ZeRO Stage 3                   │
│                                              │
│  GPU 0: Params₁ + Optim₁ + Grad₁            │
│  GPU 1: Params₂ + Optim₂ + Grad₂            │
│  → EVERYTHING sharded across GPUs            │
│  → Params gathered on-demand for forward     │
│  → Can fit models much larger than VRAM      │
└─────────────────────────────────────────────┘
```

> **Important**: ZeRO-3 adds communication overhead. Use ZeRO-2 unless you actually need the extra memory.

---

## Part 2: Model Distillation

### What is distillation?

Train a **small student model** to mimic a **large teacher model**. The student learns from the teacher's output probability distribution (soft labels), not just the correct answer:

```
Teacher (7B): "Paris" = 85%, "Lyon" = 8%, "Marseille" = 4%, ...
                                   ↓ soft labels
Student (1.2B): learns the DISTRIBUTION, not just "Paris"
```

Why soft labels work better:
- A wrong answer at 8% vs 0.1% encodes useful knowledge
- The teacher's uncertainty is informative
- More signal per training example

### The math

```
L = α × T² × KL(student/T, teacher/T) + (1-α) × CE(student, labels)
```

| Symbol | Meaning | Default |
|--------|---------|---------|
| **α** | Blend factor (0 = CE only, 1 = KL only) | 0.5 |
| **T** | Temperature (higher = softer distribution) | 2.0 |
| **KL** | Kullback-Leibler divergence | — |
| **CE** | Standard cross-entropy | — |

### Temperature effect

```
T = 1.0 (hard):  [0.90, 0.05, 0.03, 0.02]  ← peaked, mostly "Paris"
T = 2.0 (soft):  [0.60, 0.18, 0.12, 0.10]  ← smoother, more signal
T = 5.0 (very):  [0.35, 0.24, 0.22, 0.19]  ← almost uniform
```

**T = 2.0** is a good default — soft enough to transfer knowledge, hard enough to stay accurate.

### Usage

```bash
lfm-train --distill-teacher meta-llama/Llama-3.2-7B \
    --dataset sahil2801/CodeAlpaca-20k \
    --distill-temperature 2.0 \
    --distill-alpha 0.5 \
    --hub-repo your-username/lfm-distilled
```

### Python API

```python
from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",             # Student (small)
    distill_teacher="meta-llama/Llama-3.2-7B",         # Teacher (large, frozen)
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    distill_temperature=2.0,
    distill_alpha=0.5,
    use_lora=True,
    lora_r=32,
    bf16=True,
)
run_training(cfg)
```

### Distillation + DeepSpeed

For large teachers that don't fit on a single GPU, combine with ZeRO:

```python
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    distill_teacher="meta-llama/Llama-3.2-7B",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    deepspeed="zero2",          # Shard optimizer across GPUs
    distill_temperature=2.0,
    distill_alpha=0.5,
    bf16=True,
)
```

### Choosing teacher models

| Teacher | Params | Good for |
|---------|--------|----------|
| meta-llama/Llama-3.2-7B | 7B | General coding |
| Qwen/Qwen2.5-7B-Instruct | 7B | Instruction following |
| deepseek-ai/DeepSeek-Coder-6.7B | 6.7B | Code generation |
| microsoft/Phi-3-medium-4k-instruct | 14B | Reasoning (needs ZeRO-3) |

---

## Tips

1. **Start with ZeRO-2** — it has less overhead than ZeRO-3. Only use ZeRO-3 when models don't fit.
2. **Temperature 2.0–3.0** — good range for distillation. Don't go above 5.
3. **Alpha 0.5** — balanced blend. Use 0.7+ if teacher is much better than random labels.
4. **Distill on the SAME data** you'd use for SFT — distillation replaces SFT, not adds to it.
5. **Benchmark after distillation** — add `--benchmarks humaneval mbpp` to check quality.

---

## Examples

- [`deepspeed_training.py`](../examples/deepspeed_training.py) — ZeRO-2 and ZeRO-3 configs
- [`distillation.py`](../examples/distillation.py) — Distill 7B teacher → 1.2B student

## Next: Back to [README →](../README.md)
