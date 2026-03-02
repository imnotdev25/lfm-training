# 1. What is an LLM?

> **Goal**: Understand the building blocks of a language model — tokens, embeddings, attention, and transformers — with zero jargon.

---

## The Big Picture

A **Large Language Model (LLM)** is a program that predicts the next word in a sentence.

That's it. Everything else — chatbots, code generation, translation — comes from doing this one thing really well.

```
Input:  "The cat sat on the ___"
Output: "mat" (87%), "floor" (8%), "couch" (3%), ...
```

The model doesn't "understand" language — it computes probabilities.

---

## Step 1: Tokenization — Turning Text into Numbers

Computers don't understand words. They understand numbers. So the first step is converting text into a sequence of **tokens** (small pieces of text mapped to integers).

```
"Hello world" → [15496, 995]
"The cat sat"  → [464, 3797, 3332]
```

A **tokenizer** splits text into subwords. Common tokenizers:

| Tokenizer | Used by | Vocabulary size |
|-----------|---------|-----------------|
| BPE (Byte-Pair Encoding) | GPT, LLaMA | ~32,000–50,000 |
| SentencePiece | T5, LFM | ~32,000 |
| WordPiece | BERT | ~30,000 |

**Key insight**: Tokens are NOT words. Common words like "the" are one token, but rare words get split:

```
"unhappiness" → ["un", "happi", "ness"]  → [348, 43453, 1108]
```

### The Math (simple)

If our vocabulary has `V = 32,000` tokens, each token is just an integer from 0 to 31,999.

A sentence of length `n` becomes a vector: `[t₁, t₂, ..., tₙ]` where each `tᵢ ∈ {0, 1, ..., V-1}`.

---

## Step 2: Embeddings — Giving Tokens Meaning

A token ID like `3797` is meaningless. We need to convert it into a **vector** that captures its meaning.

An **embedding** maps each token to a high-dimensional vector:

```
token 3797 ("cat") → [0.12, -0.34, 0.56, 0.78, -0.91, ...] (768 dimensions)
token 2845 ("dog") → [0.14, -0.31, 0.52, 0.81, -0.88, ...] (similar to cat!)
token 1234 ("car") → [0.87, 0.23, -0.15, -0.44, 0.33, ...] (very different)
```

**Why it works**: Similar words end up close together in this vector space.

### The Math

The embedding layer is a matrix `E` of shape `[V × d]` where:
- `V` = vocabulary size (e.g., 32,000)
- `d` = embedding dimension (e.g., 768 for small models, 2048 for LFM 1.2B)

Looking up token `t` is just grabbing row `t` from the matrix:

```
embedding = E[t]  # Shape: [d]
```

For a 1.2B parameter model with `d = 2048` and `V = 32,000`:
- Embedding matrix size: 32,000 × 2,048 = **65.5 million parameters**

---

## Step 3: Attention — How the Model "Thinks"

This is the key innovation that makes modern LLMs work.

**Problem**: In "The cat sat on the mat because **it** was tired", what does "it" refer to?
- A human knows "it" = "cat"
- The model needs a mechanism to connect "it" back to "cat"

**Solution**: **Self-attention** lets every token look at every other token and decide how much to "pay attention" to it.

### Intuition

Imagine you're reading a sentence. For each word, you ask:
- "Which other words in this sentence are relevant to understanding **this** word?"
- You assign **attention weights** to each other word

```
"The cat sat on the mat because it was tired"

For the word "it":
  "cat"     → attention weight: 0.72  (very relevant!)
  "sat"     → attention weight: 0.08
  "mat"     → attention weight: 0.05
  "The"     → attention weight: 0.01
  ...
```

### The Math (Query, Key, Value)

For each token, we compute three vectors using learned weight matrices:

```
Q = X · Wq   (Query: "What am I looking for?")
K = X · Wk   (Key: "What do I contain?")
V = X · Wv   (Value: "What information do I provide?")
```

The attention scores are:

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d) · V
```

Breaking this down:
1. `Q · Kᵀ` → dot product between query and all keys (how relevant is each token?)
2. `/ √d` → scale to prevent huge numbers (d = dimension)
3. `softmax(...)` → convert to probabilities (sum to 1.0)
4. `· V` → weighted sum of values

**Example with numbers** (d=4 for simplicity):

```python
# Token "it" queries, token "cat" provides key
Q_it  = [0.5, 0.3, -0.1, 0.8]
K_cat = [0.4, 0.2, -0.2, 0.9]

# Dot product: relevance score
score = 0.5×0.4 + 0.3×0.2 + (-0.1)×(-0.2) + 0.8×0.9
      = 0.20 + 0.06 + 0.02 + 0.72
      = 1.00  ← high score! "it" should attend to "cat"
```

### Multi-Head Attention

Instead of one attention mechanism, we use **multiple heads** (e.g., 16 or 32) running in parallel:

```
Head 1: might learn syntactic relationships (subject-verb)
Head 2: might learn semantic meaning (cat → animal)
Head 3: might learn positional patterns (nearby words)
...
```

Each head has its own Q, K, V weight matrices. The outputs are concatenated.

---

## Step 4: The Transformer Block

A **transformer** stacks multiple layers, each containing:

```
┌──────────────────────────────────────┐
│  Input Embeddings                    │
├──────────────────────────────────────┤
│  Multi-Head Self-Attention           │ ← connects tokens to each other
│  + Residual Connection & Layer Norm  │
├──────────────────────────────────────┤
│  Feed-Forward Network (FFN)          │ ← processes each token independently
│  + Residual Connection & Layer Norm  │
├──────────────────────────────────────┤
│  Output → Next Layer                 │
└──────────────────────────────────────┘
```

This block is repeated `L` times (e.g., 24 layers for a 1.2B model).

### Feed-Forward Network

A simple 2-layer neural network applied to each token independently:

```
FFN(x) = GELU(x · W₁ + b₁) · W₂ + b₂
```

Where GELU is an activation function (smooth version of ReLU).

### Residual Connections

The `+` in "Residual Connection" means we **add the input back to the output**:

```
output = LayerNorm(x + Attention(x))
```

**Why?** Without residuals, gradients vanish in deep networks (information gets lost through many layers). Residuals create a "shortcut" for information to flow.

---

## Step 5: Predicting the Next Token

After passing through all transformer layers, we get a hidden state `h` for the last token. To predict the next token:

```
logits = h · Eᵀ  # Shape: [V] — one score per vocabulary token
probs = softmax(logits)  # Convert to probabilities
```

```
"The cat sat on the" → model → probs = {
    "mat": 0.12,
    "floor": 0.08,
    "couch": 0.05,
    "table": 0.04,
    ...32,000 tokens
}
```

The model then **samples** from this distribution:
- **Greedy**: always pick the highest probability (boring, repetitive)
- **Temperature sampling**: scale logits by temperature `T` before softmax
  - `T < 1.0`: more focused (less creative)
  - `T > 1.0`: more random (more creative)
- **Top-p (nucleus)**: only sample from tokens whose cumulative probability ≤ p

---

## Putting It All Together

For a 1.2B parameter model like LFM 2.5:

| Component | Size |
|-----------|------|
| Vocabulary | ~32,000 tokens |
| Embedding dim (d) | 2,048 |
| Attention heads | 16 |
| Transformer layers | 24 |
| FFN hidden dim | 5,504 |
| Total parameters | **1.2 billion** |

**Where are the 1.2 billion parameters?**

```
Embeddings:         32K × 2048     =   65M  (5%)
Attention (Q,K,V):  24 × 3 × 2048² = 302M  (25%)
Attention (output): 24 × 2048²     = 101M  (8%)
FFN:                24 × 2 × 2048 × 5504 = 541M  (45%)
Other (LayerNorm):                    ~20M  (2%)
Output head:        2048 × 32K     =   65M  (5%)
───────────────────────────────────────────
Total:                              ~1.2B
```

> **Key takeaway**: Most parameters (45%) are in the FFN layers, not attention!

---

## What Happens When You Type a Prompt

```
You type: "Write a Python function to sort a list"

1. Tokenizer splits it: [16794, 257, 11361, 2163, 284, 3297, 257, 1351]
2. Each token gets embedded: 8 vectors of dimension 2048
3. 24 transformer layers process them (attention + FFN)
4. Model predicts next token: "def" (token 4299)
5. Append "def" to input, repeat from step 2
6. Keep going until model outputs a stop token or max length
```

This is called **autoregressive generation** — each new token depends on all previous tokens.

---

## Next: [How Training Works →](02-how-training-works.md)

Now that you know what an LLM is, let's learn how we teach it to predict the right tokens.
