# 4. LoRA & Parameter-Efficient Training

> **Goal**: Understand Low-Rank Adaptation (LoRA) — why it works, the math behind it, and why you can fine-tune a 1.2B model on a free GPU.

---

## The Problem LoRA Solves

Full fine-tuning updates **all** 1.2 billion parameters. This needs:
- ~16 GB VRAM (model + gradients + optimizer states)
- Risk of catastrophic forgetting
- A separate copy of the full model for each task

**LoRA's insight**: When fine-tuning, the weight changes have **low rank** — meaning most of the change can be captured by small matrices.

---

## The Math Behind LoRA

### Weight matrices in a transformer

Each attention layer has weight matrices (Q, K, V, Output) of shape `[d × d]`.

For LFM 2.5 with `d = 2048`:
- Each weight matrix: `2048 × 2048 = 4,194,304` parameters
- 4 matrices × 24 layers = **402 million** attention parameters

### The key equation

During fine-tuning, we learn a change `ΔW` to the weight matrix:

```
W_new = W_original + ΔW
```

**LoRA's claim**: `ΔW` is low-rank. Instead of storing a full `d × d` matrix, we decompose it:

```
ΔW = A × B

Where:
  A has shape [d × r]     (2048 × 16 = 32,768 parameters)
  B has shape [r × d]     (16 × 2048 = 32,768 parameters)
  r = rank (typically 8-64, we use 16)
```

### The numbers

```
Full ΔW:     2048 × 2048 = 4,194,304 parameters per matrix
LoRA (r=16): (2048 × 16) + (16 × 2048) = 65,536 parameters per matrix

Reduction: 4,194,304 / 65,536 = 64× fewer parameters!
```

For the whole model:

```
Full fine-tuning:  ~1,200,000,000 trainable parameters
LoRA (r=16):       ~3,000,000 trainable parameters (0.25% of total)
```

### Why does this work?

Think of a `2048 × 2048` matrix as encoding 2048 independent "directions" of information. But when you fine-tune for a specific task, you only need to adjust a few key directions — the rest stays the same.

Rank `r = 16` means: "The fine-tuning change can be captured by adjusting just 16 directions out of 2048."

Visualized:

```
Full weight change ΔW (2048 × 2048):
┌────────────────────────────────────┐
│████████████████████████████████████│
│████████████████████████████████████│  4.2M parameters
│████████████████████████████████████│  (mostly redundant)
│████████████████████████████████████│
└────────────────────────────────────┘

LoRA decomposition A × B:
┌──┐   ┌────────────────────────────────────┐
│██│ × │████████████████████████████████████│
│██│   └────────────────────────────────────┘
│██│        B: [16 × 2048] = 32K params
│██│
│██│
└──┘
A: [2048 × 16] = 32K params

Total: 65K parameters (64× less!)
```

---

## How LoRA Works in Practice

### During training

```
Original model (frozen):                    LoRA adapter (trainable):
                                           
x ─→ [W_original] ─→ output_original       x ─→ [A] ─→ [B] ─→ output_lora
     (2048×2048)                                (2048×16) (16×2048)
     FROZEN ❄️                                   TRAINABLE 🔥

Final output = output_original + α × output_lora
```

Where `α` (alpha) is a scaling factor (we use α = 32 with r = 16).

The scaling `α/r` controls how much the LoRA change affects the output:
```
scaling = alpha / rank = 32 / 16 = 2.0

output = W_original × x + (scaling) × A × B × x
```

### Initialization

- **Matrix A**: initialized with small random values (Gaussian)
- **Matrix B**: initialized to **zeros**

This means at the start of training, `ΔW = A × B = 0` — the model starts exactly where the pre-trained model left off. No disruption!

---

## Memory Savings

```
┌──────────────────────────────────────────────────────┐
│ Full Fine-Tuning Memory                              │
├──────────────────────────────────────────────────────┤
│ Model (fp16):        1.2B × 2 bytes  = 2.4 GB       │
│ Gradients (fp16):    1.2B × 2 bytes  = 2.4 GB       │
│ Adam states (fp32):  1.2B × 8 bytes  = 9.6 GB       │
│ Activations:                         ≈ 2-4 GB       │
│ ─────────────────────────────────────────────        │
│ Total:                               ≈ 16-18 GB     │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│ LoRA Fine-Tuning Memory                              │
├──────────────────────────────────────────────────────┤
│ Model (fp16):        1.2B × 2 bytes  = 2.4 GB       │
│ LoRA gradients:      3M × 2 bytes    = 0.006 GB     │
│ LoRA Adam states:    3M × 8 bytes    = 0.024 GB     │
│ Activations:                         ≈ 1-2 GB       │
│ ─────────────────────────────────────────────        │
│ Total:                               ≈ 4-5 GB       │
└──────────────────────────────────────────────────────┘
```

**4× less memory!** This is why LoRA works on free Kaggle GPUs.

---

## Which Layers Get LoRA?

Not all layers benefit equally. In lfm-trainer, we apply LoRA to:

```python
target_modules = [
    "q_proj",    # Query projection (attention)
    "k_proj",    # Key projection (attention)  
    "v_proj",    # Value projection (attention)
    "o_proj",    # Output projection (attention)
    "gate_proj", # FFN gate (feed-forward)
    "up_proj",   # FFN up-projection
    "down_proj", # FFN down-projection
]
```

This covers both attention AND feed-forward layers — the two main components of each transformer block.

---

## Choosing LoRA Hyperparameters

### Rank (r)

| Rank | Trainable params | Quality | Use case |
|------|-----------------|---------|----------|
| 4 | ~800K | Lower | Quick experiments |
| 8 | ~1.5M | Good | Simple tasks |
| **16** | **~3M** | **Great** | **Default, most tasks** |
| 32 | ~6M | Very good | Complex tasks |
| 64 | ~12M | Excellent | Approaches full FT |

**Rule of thumb**: Start with `r = 16`. Increase only if quality isn't good enough.

### Alpha (α)

The scaling factor. Common practice:

```
alpha = 2 × rank

rank=16, alpha=32  ← our default
rank=8,  alpha=16
rank=32, alpha=64
```

### Dropout

Applied to LoRA layers to prevent overfitting:

```python
lora_dropout = 0.05  # 5% dropout — mild regularization
```

Higher values (0.1-0.2) for small datasets, lower (0.01-0.05) for large datasets.

---

## LoRA from Scratch (Pure Python)

Here's what LoRA does, without any library:

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """A single LoRA adapter for one weight matrix."""
    
    def __init__(self, original_layer, rank=16, alpha=32):
        super().__init__()
        d_in = original_layer.in_features
        d_out = original_layer.out_features
        
        # Freeze the original weight
        self.original = original_layer
        self.original.weight.requires_grad = False
        
        # LoRA matrices
        self.A = nn.Parameter(torch.randn(d_in, rank) * 0.01)  # small random
        self.B = nn.Parameter(torch.zeros(rank, d_out))          # zeros!
        
        self.scaling = alpha / rank
    
    def forward(self, x):
        # Original output (frozen)
        original_output = self.original(x)
        
        # LoRA output (trainable)
        lora_output = x @ self.A @ self.B * self.scaling
        
        return original_output + lora_output

# Usage:
# original = nn.Linear(2048, 2048)  # frozen pre-trained weight
# lora = LoRALayer(original, rank=16, alpha=32)
# output = lora(input_tensor)  # combines both
```

That's it! The PEFT library does this for you, but this is all that happens behind the scenes.

---

## Merging LoRA Back

After training, you can **merge** the LoRA weights back into the original:

```python
# Merge: W_final = W_original + (alpha/rank) × A × B
W_final = W_original + (32/16) * A @ B

# Now you have a single model with no LoRA overhead
# Inference speed is identical to the original model
```

This is what happens when you export to GGUF/MLX — we merge first, then quantize.

---

## Summary

| Aspect | Full Fine-Tuning | LoRA (r=16) |
|--------|-----------------|-------------|
| Trainable params | 1.2B (100%) | 3M (0.25%) |
| Memory (1.2B model) | ~16 GB | ~4 GB |
| Training speed | Slower | **3× faster** |
| Catastrophic forgetting | Higher risk | Lower risk |
| Quality | Best | 95-99% of full |
| Deployable | Replaces model | Lightweight adapter |

---

## Next: [Full Fine-Tuning →](05-full-fine-tuning.md)
