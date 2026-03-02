# 5. Full Fine-Tuning

> **Goal**: When and how to train all parameters, memory management with gradient checkpointing, and practical tips.

---

## When to Use Full Fine-Tuning

| Choose **Full Fine-Tuning** when | Choose **LoRA** when |
|----------------------------------|---------------------|
| You need maximum quality | Good enough is fine |
| You have enough VRAM (≥16 GB) | Limited VRAM (4-8 GB) |
| Your dataset is large (50K+ examples) | Small datasets (<50K) |
| You want a single merged model | You want swappable adapters |
| You're doing a final production run | You're experimenting |

---

## Memory Requirements

For a 1.2B parameter model:

```
Component              fp32         fp16/bf16
──────────────────────────────────────────────
Model weights          4.8 GB       2.4 GB
Gradients              4.8 GB       2.4 GB
Adam (m + v)           9.6 GB       9.6 GB (always fp32)
Activations            2-6 GB       1-3 GB
──────────────────────────────────────────────
Total                  ~21 GB       ~16 GB
```

### GPU compatibility

| GPU | VRAM | Full FT (1.2B)? | Batch size |
|-----|------|------------------|------------|
| Kaggle P100 | 16 GB | ✅ (tight) | 1 |
| Kaggle 2× T4 | 2× 16 GB | ✅ | 2 |
| Colab T4 | 15 GB | ⚠️ (very tight) | 1 |
| RTX 3090 | 24 GB | ✅ | 2-4 |
| A100 40GB | 40 GB | ✅ | 8 |

---

## Gradient Checkpointing — The Memory Trick

### The problem

During the forward pass, we save **activations** (intermediate results) at every layer because we need them for backpropagation:

```
Forward pass saves:
Layer 1 output → stored in VRAM (300 MB)
Layer 2 output → stored in VRAM (300 MB)
...
Layer 24 output → stored in VRAM (300 MB)

Total activation memory: ~3-6 GB
```

### The solution

**Gradient checkpointing** trades compute for memory: instead of saving all activations, we save only a few "checkpoints" and recompute the rest during backpropagation.

```
Without checkpointing:
Forward:  Save ████████████████████████ (24 layers = 6 GB)
Backward: Reuse saved activations

With checkpointing:
Forward:  Save ██░░░░██░░░░██░░░░██░░░░ (6 checkpoints = 1.5 GB)
Backward: Recompute ░░ layers from nearest checkpoint
```

**Trade-off**: ~33% slower training, but ~50-70% less activation memory.

### In lfm-trainer

Gradient checkpointing is automatically enabled when you use `--full-finetune`:

```python
# This happens inside trainer.py
model.gradient_checkpointing_enable()
```

---

## Practical Tips for Full Fine-Tuning

### 1. Use a lower learning rate

```python
# LoRA can handle higher LR (adapters are small)
cfg = TrainingConfig(use_lora=True, learning_rate=2e-4)   # aggressive

# Full FT needs gentler updates (changing everything!)
cfg = TrainingConfig(use_lora=False, learning_rate=5e-5)  # conservative
```

**Why?** With full fine-tuning, you're modifying the model's core knowledge. Large steps can destroy important information (catastrophic forgetting).

### 2. Use bf16, not fp16

```python
cfg = TrainingConfig(bf16=True, fp16=False)
```

**Why?** bf16 has a larger numerical range than fp16, which prevents overflow during training. fp16 can produce NaN values more easily.

```
fp16 range: ±65,504  (can overflow with large gradients)
bf16 range: ±3.4×10³⁸ (same as fp32, safe)
```

### 3. Smaller batch, more accumulation

```python
cfg = TrainingConfig(
    per_device_train_batch_size=1,      # fits in memory
    gradient_accumulation_steps=8,      # effective batch = 8
)
```

### 4. Monitor eval loss for overfitting

```python
cfg = TrainingConfig(
    eval_split=0.1,  # hold out 10% for validation
)
```

If eval loss starts rising while train loss keeps falling → you're overfitting → stop early or reduce epochs.

---

## Behind the Scenes: Full FT vs LoRA

```
Full Fine-Tuning:
┌──────────────────────┐
│ Load model (2.4 GB)  │
│ All params trainable │ ← ALL 1.2B weights get gradients
│ Adam tracks ALL      │ ← 9.6 GB for optimizer states
│ Save entire model    │ ← 2.4 GB checkpoint
└──────────────────────┘

LoRA:
┌──────────────────────┐
│ Load model (2.4 GB)  │
│ Freeze base ❄️       │ ← 1.2B weights are frozen
│ Add LoRA adapters 🔥  │ ← Only 3M weights train
│ Adam tracks 3M only  │ ← 24 MB for optimizer
│ Save adapter only    │ ← 12 MB checkpoint
└──────────────────────┘
```

---

## Summary

| Aspect | Full Fine-Tuning |
|--------|-----------------|
| Quality | Best possible |
| Memory | ~16 GB for 1.2B |
| Speed | Slower (gradient checkpointing) |
| Learning rate | Low (5e-5) |
| Precision | bf16 recommended |
| Risk | Higher catastrophic forgetting |
| Output | Complete model (no merging needed) |

---

## Next: [Data Preparation →](06-data-preparation.md)
