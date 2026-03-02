# 2. How Training Works

> **Goal**: Understand the math behind training — loss functions, backpropagation, and gradient descent — explained from scratch.

---

## The Core Idea

Training a language model means:

1. Show the model a sentence
2. Ask it to predict the next token
3. Measure how wrong it was (the **loss**)
4. Adjust its parameters to be less wrong
5. Repeat billions of times

---

## Step 1: The Loss Function

We need to measure "how wrong" the model is. For language models, we use **cross-entropy loss**.

### Intuition

If the correct next token is "mat" and the model predicts:

```
Model output (probabilities):
  "mat"   → 0.12  (12% — should be close to 100%!)
  "floor" → 0.08
  "cat"   → 0.03
  ...
```

The loss measures: **"How surprised are we that the correct answer got such a low probability?"**

### The Math

For one prediction, the cross-entropy loss is:

```
L = -log(p_correct)
```

Where `p_correct` is the probability the model assigns to the correct token.

**Examples:**

| Model's confidence in correct answer | Loss | Interpretation |
|--------------------------------------|------|----------------|
| 0.99 (99%) | -log(0.99) = 0.01 | Great! Almost no loss |
| 0.50 (50%) | -log(0.50) = 0.69 | Mediocre — uncertain |
| 0.12 (12%) | -log(0.12) = 2.12 | Bad — very wrong |
| 0.01 (1%) | -log(0.01) = 4.61 | Terrible — clueless |

**Key property**: The loss is always ≥ 0, and only equals 0 when the model is 100% confident in the right answer.

### Over a whole sequence

For a sentence with `n` tokens, the total loss is the average:

```
L_total = -(1/n) × Σ log(p(tᵢ | t₁, t₂, ..., tᵢ₋₁))
```

This says: "For each token in the sentence, measure how well the model predicted it given all previous tokens."

### Perplexity

You'll often see **perplexity** reported instead of loss:

```
Perplexity = e^(loss)
```

| Perplexity | Meaning |
|-----------|---------|
| 1.0 | Perfect — model is never surprised |
| 10.0 | Model is choosing between ~10 equally likely tokens |
| 100.0 | Model is very confused |
| 1000.0 | Model knows nothing |

A well-trained 1.2B coding model typically achieves perplexity ~5-15 on code.

---

## Step 2: Backpropagation — Finding Who's Responsible

Now we know how wrong the model is (the loss). But the model has **1.2 billion parameters** — which ones should we change, and by how much?

**Backpropagation** answers this by computing the **gradient** of the loss with respect to each parameter.

### What is a gradient?

A gradient tells you: **"If I increase this parameter by a tiny amount, how much does the loss change?"**

```
∂L/∂w = "how much does loss change when parameter w changes?"
```

- If `∂L/∂w > 0`: increasing `w` increases the loss → we should decrease `w`
- If `∂L/∂w < 0`: increasing `w` decreases the loss → we should increase `w`
- If `∂L/∂w ≈ 0`: this parameter doesn't matter much right now

### The Chain Rule

A transformer has many layers, and each layer depends on the previous one:

```
Input → Layer 1 → Layer 2 → ... → Layer 24 → Loss
```

To find how Layer 1's parameters affect the loss, we use the **chain rule**:

```
∂L/∂w₁ = (∂L/∂output) × (∂output/∂layer24) × ... × (∂layer2/∂layer1) × (∂layer1/∂w₁)
```

We multiply the gradients backward through the network — hence "back-propagation."

### In practice

You never compute gradients by hand. PyTorch does it automatically:

```python
loss = model(input_tokens, labels=target_tokens).loss  # forward pass
loss.backward()  # backward pass — computes all gradients
# Now every parameter has a .grad attribute
```

---

## Step 3: Gradient Descent — Updating Parameters

Now we have a gradient for every parameter. Time to update them.

### Simple Gradient Descent

```
w_new = w_old - learning_rate × gradient
```

The **learning rate** (η, typically 1e-4 to 5e-5) controls step size:

```
Too large (η = 0.1):  → overshoots, loss explodes
Too small (η = 1e-8): → barely moves, training takes forever  
Just right (η = 2e-4): → steady improvement
```

### Adam Optimizer (what we actually use)

Simple gradient descent has problems:
- Some parameters need bigger steps, others smaller
- Gradients can be noisy

**Adam** fixes this by tracking:
1. **Momentum** (m): running average of gradients (smooths out noise)
2. **Variance** (v): running average of squared gradients (adapts step size per parameter)

```
m_t = 0.9 × m_{t-1} + 0.1 × gradient        # smooth gradient
v_t = 0.999 × v_{t-1} + 0.001 × gradient²    # gradient magnitude
w_new = w_old - lr × m_t / (√v_t + ε)         # update
```

**Why Adam is better**: Parameters with consistently large gradients get smaller steps (already learning fast). Parameters with small gradients get larger steps (need more push).

---

## Step 4: Batches and Epochs

### Batches

We don't update after every single sentence. Instead, we process a **batch** of sentences at once:

```
Batch size = 4 (process 4 sentences at once)

Sentence 1 → loss₁ = 2.3
Sentence 2 → loss₂ = 1.8
Sentence 3 → loss₃ = 3.1
Sentence 4 → loss₄ = 2.0

Batch loss = (2.3 + 1.8 + 3.1 + 2.0) / 4 = 2.3
→ Update all parameters once
```

**Why batches?**
- More stable gradients (average over multiple examples)
- GPU parallelism (process 4 sentences simultaneously)

### Gradient Accumulation

When the GPU doesn't have enough memory for large batches:

```
Effective batch size = batch_size × gradient_accumulation_steps

batch_size = 2, gradient_accumulation = 4
→ Effective batch size = 8

Step 1: Process 2 sentences, accumulate gradients (don't update yet)
Step 2: Process 2 more, accumulate
Step 3: Process 2 more, accumulate
Step 4: Process 2 more, NOW update parameters
```

### Epochs

One **epoch** = one pass through the entire dataset.

```
Dataset: 10,000 examples
Batch size: 4
Steps per epoch: 10,000 / 4 = 2,500

Training for 3 epochs = seeing each example 3 times = 7,500 total steps
```

For fine-tuning, **1-3 epochs** is usually enough. More risks overfitting.

---

## Step 5: Learning Rate Scheduling

The learning rate shouldn't stay constant. Common strategies:

### Warmup + Cosine Decay

```
Learning Rate
    ↑
0.0002|        ╱‾‾‾‾‾‾‾‾‾╲
      |       ╱             ╲
      |      ╱                ╲
      |     ╱                   ╲
0.0000|____╱                     ╲___
      └──────────────────────────────→ Training Steps
      |warm| ← training →        |end|
      | up |                      |   |
```

1. **Warmup** (first ~3% of steps): gradually increase LR from 0 to max
   - Prevents wild weight updates when the model hasn't adapted yet
2. **Cosine decay**: slowly decrease LR to near 0
   - Fine adjustments at the end, coarse adjustments at the start

---

## What Happens Behind the Scenes (One Training Step)

```
1. Load batch of 4 sentences
2. Tokenize them: "def hello():" → [4299, 23748, 33529, 25]
3. Forward pass through 24 transformer layers:
   └→ For each token position, predict the next token
   └→ Compare prediction to actual next token
   └→ Compute cross-entropy loss
4. Backward pass:
   └→ Compute gradient of loss w.r.t. all 1.2B parameters
   └→ This takes about 2× the compute of the forward pass
5. Optimizer step:
   └→ Adam updates all 1.2B parameters
   └→ Clear gradients for next step
6. Log loss value, check if we should save a checkpoint
7. Repeat
```

### Memory during training

For a 1.2B model in fp16:

```
Model parameters:    1.2B × 2 bytes  = 2.4 GB
Gradients:           1.2B × 2 bytes  = 2.4 GB  
Adam optimizer state: 1.2B × 8 bytes = 9.6 GB (m + v, in fp32)
Activations:         ~2-4 GB (depends on batch size & seq length)
────────────────────────────────────────────
Total:               ~16-18 GB for full fine-tuning
```

This is why full fine-tuning of even a 1.2B model needs a good GPU!

And why LoRA (next chapter) is so important — it dramatically reduces this.

---

## Summary

| Concept | What it does | The math |
|---------|-------------|----------|
| Loss | Measures how wrong the model is | `L = -log(p_correct)` |
| Backprop | Finds which params caused the error | Chain rule: `∂L/∂w` |
| Gradient descent | Updates params to reduce error | `w -= lr × ∂L/∂w` |
| Adam | Smarter updates with momentum | `w -= lr × m/(√v + ε)` |
| Batch size | Number of examples per update | Larger = more stable |
| Epoch | One pass through all data | 1-3 for fine-tuning |
| Learning rate | How big each update step is | 1e-4 to 5e-5 typical |

---

## Next: [Fine-Tuning vs Training from Scratch →](03-fine-tuning-explained.md)
