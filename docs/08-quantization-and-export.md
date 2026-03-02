# 8. Quantization & Export

> **Goal**: Understand what quantization is, how GGUF and MLX work, and why a 4-bit model can run on your laptop.

---

## What is Quantization?

Your trained model stores each parameter as a 16-bit floating-point number (fp16). **Quantization** converts these to smaller representations:

```
fp16 (16 bits):   ±65,504 range,  high precision
int8  (8 bits):   -128 to 127,    good precision
int4  (4 bits):   -8 to 7,        lower precision but 4× smaller
```

### The trade-off

```
Precision:  fp16 > int8 > int6 > int4
Size:       fp16 > int8 > int6 > int4 (smaller is better)
Speed:      int4 > int6 > int8 > fp16 (smaller is faster)
Quality:    fp16 > int8 > int6 > int4 (higher is better)
```

### Size comparison (1.2B model)

| Format | Bits | Model size | RAM needed |
|--------|------|-----------|------------|
| fp16 | 16 | 2.4 GB | ~3 GB |
| int8 (Q8_0) | 8 | 1.2 GB | ~1.5 GB |
| int6 (Q6_K) | 6 | 0.9 GB | ~1.2 GB |
| int4 (Q4_K_M) | 4 | 0.7 GB | ~1.0 GB |

A 4-bit quantized 1.2B model is only **700 MB** — small enough to run on a phone!

---

## How Quantization Works (The Math)

### Linear Quantization (simplified)

To convert a float to an integer:

```
1. Find the range: min_val = -0.5, max_val = 1.2
2. Compute scale: scale = (max_val - min_val) / (2^bits - 1)
   For 8-bit: scale = (1.2 - (-0.5)) / 255 = 0.00667
3. Compute zero point: zero = round(-min_val / scale) = 75
4. Quantize: q = round(float_val / scale) + zero
5. Dequantize: float_val ≈ (q - zero) × scale
```

**Example:**

```
Original value: 0.75

Quantize (8-bit):
  q = round(0.75 / 0.00667) + 75 = round(112.4) + 75 = 187

Dequantize:
  float = (187 - 75) × 0.00667 = 112 × 0.00667 = 0.747

Error: |0.75 - 0.747| = 0.003  (tiny!)
```

### Block Quantization (what GGUF uses)

Instead of one scale for all values, GGUF groups weights into **blocks** (typically 32 values) and computes a scale per block:

```
Block 1: [0.1, 0.3, -0.2, 0.5, ...]  → scale₁ = 0.004
Block 2: [1.2, -0.8, 0.9, 2.1, ...]  → scale₂ = 0.012
...
```

This is more accurate because each block has its own range.

---

## GGUF Format

**GGUF** (GPT-Generated Unified Format) is the standard for running quantized models with [llama.cpp](https://github.com/ggerganov/llama.cpp).

### Quantization levels

| Name | Description | Quality | Size ratio |
|------|------------|---------|------------|
| Q4_K_M | 4-bit with medium block size | Good | 0.29× |
| Q6_K | 6-bit with large block size | Very good | 0.44× |
| Q8_0 | 8-bit, simple rounding | Near-lossless | 0.54× |

**K** = "K-quant" (smarter block-based method)
**M** = "Medium" block size (balanced)

### What lfm-trainer exports

```python
cfg = TrainingConfig(export_gguf=True)
```

This creates three versions:
```
your-model-GGUF-Q4_K_M/  → Smallest, best for laptops
your-model-GGUF-Q6_K/    → Balanced
your-model-GGUF-Q8_0/    → Best quality, still compressed
```

All three are pushed to separate HuggingFace repos with shared version tags.

---

## MLX Format (Apple Silicon)

**MLX** is Apple's machine learning framework, optimized for M1/M2/M3 chips.

### Why MLX?

Apple Silicon has **unified memory** — the GPU and CPU share the same RAM. MLX exploits this for:
- Zero-copy data transfer (no GPU↔CPU bottleneck)
- Efficient inference on Mac

### Quantization levels

| Name | Bits | Quality |
|------|------|---------|
| 4-bit | 4 | Good for chat |
| 6-bit | 6 | Very good |
| 8-bit | 8 | Best quality |

### What lfm-trainer exports

```python
cfg = TrainingConfig(export_mlx=True)
```

Creates three versions:
```
your-model-MLX-4bit/
your-model-MLX-6bit/
your-model-MLX-8bit/
```

---

## The Export Pipeline

```
Step 1: Merge LoRA → Full Model
┌──────────────┐     ┌──────────────────┐
│ Base model    │ +   │ LoRA adapter     │ = Merged model
│ (1.2B frozen) │     │ (3M trained)     │   (1.2B updated)
└──────────────┘     └──────────────────┘

Step 2: Quantize
┌──────────────────┐     ┌─────────────────────┐
│ Merged model     │ ──→ │ Q4_K_M (0.7 GB)     │
│ (2.4 GB, fp16)   │ ──→ │ Q6_K   (0.9 GB)     │
│                  │ ──→ │ Q8_0   (1.2 GB)     │
└──────────────────┘     └─────────────────────┘

Step 3: Push to Hub
  your-name/model-GGUF-Q4_K_M  (tagged v20260302-123456)
  your-name/model-GGUF-Q6_K    (tagged v20260302-123456)
  your-name/model-GGUF-Q8_0    (tagged v20260302-123456)
```

### Why merge first?

LoRA stores only the adapter (12 MB). But GGUF needs a complete model. So we:
1. Load the base model
2. Add LoRA: `W_final = W_base + (α/r) × A × B`
3. Save the merged weights
4. Quantize the merged model

For **full fine-tuning**, no merge is needed (the model already has all weights).

---

## Quality Impact of Quantization

How much quality do you lose? For a 1.2B coding model:

```
fp16 (baseline): HumanEval pass@1 = 25.0%
Q8_0:            HumanEval pass@1 = 24.8%  (-0.2%)  ← nearly identical
Q6_K:            HumanEval pass@1 = 24.2%  (-0.8%)  ← very good
Q4_K_M:          HumanEval pass@1 = 22.5%  (-2.5%)  ← small quality drop
```

For most use cases, **Q4_K_M is good enough** and runs ~3× faster than fp16.

---

## Running Quantized Models

### GGUF with llama.cpp

```bash
# Install
brew install llama.cpp  # Mac
# or build from source

# Run
llama-cli -m your-model-Q4_K_M.gguf -p "Write a Python function to"
```

### MLX on Mac

```bash
pip install mlx-lm
mlx_lm.generate --model your-model-MLX-4bit --prompt "Write a Python function to"
```

### With Ollama

```bash
# Create a Modelfile
echo 'FROM ./your-model-Q4_K_M.gguf' > Modelfile
ollama create mymodel -f Modelfile
ollama run mymodel "Write a Python function to sort a list"
```

---

## Summary

| Format | Best for | Size (1.2B) | Quality loss |
|--------|---------|-------------|-------------|
| fp16 | Training, reference | 2.4 GB | None |
| GGUF Q8_0 | High-quality local inference | 1.2 GB | ~0.2% |
| GGUF Q6_K | Balanced | 0.9 GB | ~0.8% |
| GGUF Q4_K_M | Laptops, phones, speed | 0.7 GB | ~2.5% |
| MLX 4-bit | Apple Silicon | 0.7 GB | ~2.5% |

---

## Next: [Architecture Deep-Dive →](09-architecture-deep-dive.md)
