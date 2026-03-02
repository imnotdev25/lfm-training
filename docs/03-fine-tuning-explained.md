# 3. Fine-Tuning vs Training from Scratch

> **Goal**: Understand why we fine-tune instead of training from scratch, what transfer learning is, and how to avoid catastrophic forgetting.

---

## Why Not Train from Scratch?

Training a model from scratch means:
1. Start with random parameters (the model knows nothing)
2. Train on **trillions** of tokens for weeks/months
3. Requires enormous compute ($100K–$10M+)

| Model | Training cost (estimated) | Training compute |
|-------|--------------------------|------------------|
| GPT-3 (175B) | ~$5M | 3,640 petaflop-days |
| LLaMA-2 (70B) | ~$2M | 1,720,320 GPU-hours |
| LFM 2.5 (1.2B) | ~$50K | Thousands of GPU-hours |
| Your fine-tune (1.2B LoRA) | **~$1-5** | ~30 min on Kaggle |

**Fine-tuning** means: take a model that already knows language and code, and teach it your specific style/task.

---

## Transfer Learning — Standing on the Shoulders of Giants

The idea behind fine-tuning is **transfer learning**:

```
Pre-training (someone else did this):
  "Read the entire internet, learn what code looks like"
  → Result: a model that can generate plausible code
  
Fine-tuning (you do this):  
  "Now learn THIS specific style of code / tool-calling format"
  → Result: a model specialized for your use case
```

### Why it works

During pre-training, the model learns:
- **Layer 1-8**: Basic syntax, token patterns, simple structures
- **Layer 9-16**: Code semantics, function patterns, logic flow
- **Layer 17-24**: High-level reasoning, style, task-specific behavior

When you fine-tune, the **lower layers barely change** (they already know syntax). The **upper layers adapt** to your specific format.

```
Layer 24 ████████████  ← changes a lot (task-specific)
Layer 20 ██████░░░░░░  ← moderate changes
Layer 16 ████░░░░░░░░  ← small changes
Layer 12 ██░░░░░░░░░░  ← tiny changes
Layer 8  █░░░░░░░░░░░  ← almost no change
Layer 4  ░░░░░░░░░░░░  ← frozen (syntax is universal)
Layer 1  ░░░░░░░░░░░░  ← frozen
```

This is why fine-tuning is so efficient — you're only updating ~5-10% of the knowledge.

---

## What Changes During Fine-Tuning

### Before fine-tuning (base model)
```
User: Write a function to sort a list
Model: def sort_list(lst):
           return sorted(lst)  # generic, basic response
```

### After fine-tuning on coding dataset
```
User: Write a function to sort a list
Model: def sort_list(lst: list[int]) -> list[int]:
           """Sort a list of integers in ascending order.
           
           Args:
               lst: The list to sort
               
           Returns:
               A new sorted list
           
           Examples:
               >>> sort_list([3, 1, 2])
               [1, 2, 3]
           """
           return sorted(lst)
```

The fine-tuned model learned:
- Type hints
- Docstrings
- Examples
- Consistent formatting

---

## Catastrophic Forgetting

The biggest risk in fine-tuning: the model **forgets** what it knew.

```
Before fine-tuning: Can write Python, JavaScript, Rust, explain concepts
After bad fine-tuning: Only writes Python tool-calling, forgot everything else
```

### Why it happens

If you only train on one narrow dataset (e.g., tool-calling), the gradients keep pushing the model toward that task. The parameters that encoded other knowledge get overwritten.

### How to prevent it

| Strategy | How it works | Our tool |
|----------|-------------|----------|
| **LoRA** | Only change a small subset of parameters | `use_lora=True` (default) |
| **Low learning rate** | Small steps = gentle changes | `learning_rate=5e-5` |
| **Few epochs** | Don't overtrain | `num_train_epochs=1-3` |
| **Diverse data** | Mix multiple datasets | `dataset_paths=[ds1, ds2]` |
| **Eval split** | Monitor overfitting | `eval_split=0.1` |

---

## Types of Fine-Tuning

### 1. Full Fine-Tuning
- Update **all** parameters
- Best quality, but needs most memory
- Higher risk of catastrophic forgetting
- Use for: maximum performance when you have enough data

```python
cfg = TrainingConfig(use_lora=False)
```

### 2. LoRA (Low-Rank Adaptation)
- Only train small **adapter** matrices (~1-5% of params)
- Fast, memory-efficient, easy to swap between tasks
- Slightly lower quality than full fine-tuning
- Use for: most cases, especially limited hardware

```python
cfg = TrainingConfig(use_lora=True, lora_r=16)
```

### 3. Continual Fine-Tuning
- Start from a previously fine-tuned model
- Stack multiple rounds of specialization
- Use for: building on previous work

```python
cfg = TrainingConfig(resume_from_model="user/lfm-v1")
```

### Comparison

```
Full Fine-Tuning:     ████████████████████  Quality
                      ████████████████████  Memory
                      ██░░░░░░░░░░░░░░░░░░  Forgetting Risk (high)

LoRA:                 ████████████████░░░░  Quality
                      ████░░░░░░░░░░░░░░░░  Memory
                      ██████████████████░░  Forgetting Risk (low)

Continual:            ████████████████████  Quality (cumulative)
                      ████░░░░░░░░░░░░░░░░  Memory
                      ████████████████░░░░  Forgetting Risk (moderate)
```

---

## Behind the Scenes: What lfm-trainer Does

When you run `run_training(cfg)`, here's the full sequence:

```
1. Load base model (liquid/LFM2.5-1.2B-Base)
   → 1.2B parameters loaded in fp16 (2.4 GB)

2. Apply LoRA adapters (or skip if full fine-tuning)
   → Adds ~3M trainable parameters (0.25% of total)

3. Load and format dataset
   → Convert to chat template: [USER, ASSISTANT] turns
   → Optional: quality filter, tool-calling filter

4. Configure training
   → Set up optimizer (Adam), scheduler (cosine warmup)
   → Set up SFTTrainer from the trl library

5. Train
   → For each batch:
     a. Forward pass (compute predictions)
     b. Compute loss (cross-entropy)
     c. Backward pass (compute gradients)
     d. Optimizer step (update parameters)
   → Save checkpoints every N steps

6. Save & push
   → Save adapter weights (or full model)
   → Push to HuggingFace Hub

7. Benchmark (optional)
   → Run HumanEval, MBPP, etc.
   → Generate model card with results

8. Export (optional)
   → Merge LoRA into base model
   → Quantize to GGUF / MLX
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| Pre-training | Expensive ($50K+), teaches general knowledge |
| Fine-tuning | Cheap ($1-5), specializes for your task |
| Transfer learning | Reuse knowledge from pre-training |
| Catastrophic forgetting | Model forgets old knowledge during fine-tuning |
| LoRA | Efficient: only trains ~1% of parameters |
| Full fine-tuning | Best quality but needs more memory |

---

## Next: [LoRA & Parameter-Efficient Training →](04-lora-explained.md)
