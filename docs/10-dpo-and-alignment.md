# 10. DPO, PPO, GRPO & Preference Alignment

> **Goal**: Understand how to align LLMs with human preferences after SFT using DPO, PPO, or GRPO.

---

## Why SFT Isn't Enough

Supervised Fine-Tuning (SFT) teaches the model **what** to say — it learns to mimic the training data. But it doesn't learn **which** of two valid responses is *better*.

```
Prompt: "Write a function to check if a number is prime"

Response A (clean, documented):
  def is_prime(n):
      """Check primality."""
      if n < 2: return False
      for i in range(2, int(n**0.5) + 1):
          if n % i == 0: return False
      return True

Response B (works but ugly):
  def f(x):
      if x<2:return 0
      for i in range(2,x):
          if x%i==0:return 0
      return 1
```

Both are correct. SFT can't distinguish quality. **Alignment methods can.**

---

## Three Alignment Methods

| Method | What it needs | Complexity | Stability | Best for |
|--------|--------------|------------|-----------|----------|
| **DPO** | Preference pairs (chosen/rejected) | Simple | Very stable | Most use cases |
| **PPO** | Reward model + prompts | Complex (3 models) | Can be unstable | Maximum quality |
| **GRPO** | Reward function + prompts | Medium | Stable | Custom reward signals |

### Quick decision guide:

```
Want simple, stable alignment?  →  DPO
Have a reward model?            →  PPO
Want custom scoring logic?      →  GRPO
Not sure?                       →  Start with DPO
```

---

## 1. DPO — Direct Preference Optimization

### How it works

DPO learns from **pairs** of responses — one "chosen" (better) and one "rejected" (worse):

```
Training data:
  Prompt:   "Write a prime checker"
  Chosen:    Clean, documented code      ← prefer this
  Rejected:  Messy, unclear code         ← avoid this
```

### The math

```
L_DPO = -E[log σ(β × (log π(y_w|x)/π_ref(y_w|x) - log π(y_l|x)/π_ref(y_l|x)))]

Where:
  y_w = chosen (winning) response
  y_l = rejected (losing) response
  π = current model
  π_ref = reference model (frozen SFT copy)
  β = temperature (controls conservatism)
  σ = sigmoid function
```

**In plain English**: "Increase probability of chosen responses, decrease rejected, but don't drift too far from the SFT model."

### The β parameter

```
β = 0.05  → aggressive (changes more from SFT)
β = 0.1   → balanced (default, recommended)
β = 0.5   → conservative (stays close to SFT)
```

### Usage

```python
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    alignment_method="dpo",
    alignment_dataset="argilla/dpo-mix-7k",
    dpo_beta=0.1,
)
run_training(cfg)
```

```bash
lfm-train --dataset sahil2801/CodeAlpaca-20k \
    --alignment-method dpo \
    --alignment-dataset argilla/dpo-mix-7k \
    --dpo-beta 0.1
```

---

## 2. PPO — Proximal Policy Optimization (Classic RLHF)

### How it works

PPO is the **original RLHF** approach (used by ChatGPT, Claude). It uses a separate reward model to score responses:

```
┌─────────────┐    generate    ┌──────────┐    score    ┌──────────────┐
│ Policy Model │ ─────────────→│ Response │ ──────────→│ Reward Model │
│   (trainable)│               └──────────┘            │  (frozen)    │
└──────┬───────┘                                       └──────┬───────┘
       │                                                      │
       │              ┌─────────┐                             │
       └──────────────│ PPO     │←────────── reward score ────┘
                      │ Update  │
                      └─────────┘
```

### The math

PPO uses a clipped objective to prevent too-large updates:

```
L_PPO = -E[min(r_t × A_t, clip(r_t, 1-ε, 1+ε) × A_t)]

Where:
  r_t = π(a_t|s_t) / π_old(a_t|s_t)  (probability ratio)
  A_t = advantage (how much better than expected)
  ε = clipping range (usually 0.2)
```

**In plain English**: "Update the model to increase probability of high-reward responses, but clip updates to prevent instability."

### The RLHF pipeline (3 models!)

```
Model 1: Policy (the model being trained)
Model 2: Reference (frozen copy, prevents drift)
Model 3: Reward model (scores responses, also frozen)
```

This is why PPO needs ~3× more memory than DPO.

### Usage

```python
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    alignment_method="ppo",
    alignment_dataset="Anthropic/hh-rlhf",
    reward_model="OpenAssistant/reward-model-deberta-v3-large-v2",
    alignment_learning_rate=1e-5,
    alignment_max_steps=200,
)
run_training(cfg)
```

```bash
lfm-train --dataset sahil2801/CodeAlpaca-20k \
    --alignment-method ppo \
    --alignment-dataset Anthropic/hh-rlhf \
    --reward-model OpenAssistant/reward-model-deberta-v3-large-v2
```

---

## 3. GRPO — Group Relative Policy Optimization

### How it works

GRPO is **DeepSeek's innovation** (used in DeepSeek-R1). Instead of a reward model, it uses **reward functions** and generates **multiple completions** per prompt:

```
Prompt → Generate 4 responses → Score all with reward function → Use group ranking to update

Response 1: "def is_prime(n): ..."  → score: 0.9  ← best in group
Response 2: "def f(x): ..."         → score: 0.3
Response 3: "# todo"                → score: 0.1  ← worst in group
Response 4: "def check(n): ..."     → score: 0.7
```

The model learns from the **relative ranking within the group**, not absolute scores.

### The math

```
L_GRPO = -E[1/G Σ min(r_i × A_i, clip(r_i, 1-ε, 1+ε) × A_i) - β × KL(π || π_ref)]

Where:
  G = group size (number of generations per prompt)
  A_i = (reward_i - mean(rewards)) / std(rewards)  ← normalized within group
  r_i = π(y_i|x) / π_old(y_i|x)
  β × KL = penalizes drifting from reference model
```

**Key insight**: GRPO normalizes rewards **within each group**, so absolute reward values don't matter — only relative ranking does. This makes it robust to poorly calibrated reward functions.

### Why GRPO is exciting

1. **No reward model needed** — just a Python function
2. **Simpler than PPO** — no value head, no GAE
3. **Naturally explores** — multiple generations surface diverse solutions
4. **Proven at scale** — powers DeepSeek-R1

### Built-in reward functions

lfm-trainer includes reward functions you can use out of the box:

```python
from lfm_trainer.dpo import (
    _default_code_reward_fn,     # Code quality heuristics
    code_correctness_reward,     # Syntax validity checker
    length_and_quality_reward,   # Conciseness + structure
)
```

Or write your own:

```python
def my_reward(completions: list[str], **kwargs) -> list[float]:
    """Custom reward: prefer responses that include docstrings."""
    return [1.0 if '"""' in c else 0.0 for c in completions]
```

### Usage

```python
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    alignment_method="grpo",
    alignment_dataset="argilla/dpo-mix-7k",
    grpo_num_generations=4,
    alignment_learning_rate=5e-5,
)
run_training(cfg)
```

```bash
lfm-train --dataset sahil2801/CodeAlpaca-20k \
    --alignment-method grpo \
    --alignment-dataset argilla/dpo-mix-7k \
    --grpo-generations 4
```

For custom reward functions, use the Python API:

```python
from lfm_trainer.dpo import run_grpo

def docstring_reward(completions, **kwargs):
    return [1.0 if '"""' in c else 0.0 for c in completions]

run_grpo(cfg, reward_fn=docstring_reward)
```

---

## Recommended Datasets

### For DPO (preference pairs)

| Dataset | Size | Best for | Link |
|---------|------|----------|------|
| **argilla/dpo-mix-7k** | 7K | Coding models | [HuggingFace](https://huggingface.co/datasets/argilla/dpo-mix-7k) |
| **Anthropic/hh-rlhf** | 170K | Safety + helpfulness | [HuggingFace](https://huggingface.co/datasets/Anthropic/hh-rlhf) |
| **yitingxie/rlhf-reward-datasets** | 76K | General alignment | [HuggingFace](https://huggingface.co/datasets/yitingxie/rlhf-reward-datasets) |
| Intel/orca_dpo_pairs | 12K | Orca-style reasoning | HuggingFace |
| HuggingFaceH4/ultrafeedback_binarized | 60K | GPT-4 judged | HuggingFace |

### For PPO (prompts + reward model)

| Dataset | Use with |
|---------|----------|
| Anthropic/hh-rlhf | OpenAssistant/reward-model-deberta-v3-large-v2 |
| yitingxie/rlhf-reward-datasets | Any reward model |

### For GRPO (prompts + reward function)

Use any dataset with a "prompt" column. The reward function scores the generated responses — no labels needed.

### Dataset format requirements

```
DPO needs:  prompt, chosen, rejected
PPO needs:  prompt (+ reward model)
GRPO needs: prompt (+ reward function)
```

lfm-trainer auto-detects column name variations:
- `prompt` / `question` / `instruction` / `input` / `query` / `text`
- `chosen` / `preferred` / `accepted` / `positive`
- `rejected` / `dispreferred` / `negative` / `refused`

---

## Comparison: DPO vs PPO vs GRPO

| Aspect | DPO | PPO | GRPO |
|--------|-----|-----|------|
| **Data needed** | Preference pairs | Prompts + reward model | Prompts + reward function |
| **Models in memory** | 2 (policy + reference) | 3 (policy + ref + reward) | 2 (policy + reference) |
| **Training stability** | Very stable | Can be unstable | Stable |
| **Compute cost** | Low (same as SFT) | High (3× SFT) | Medium (generate multiple) |
| **Quality ceiling** | Very good | Highest (in theory) | Very good |
| **Custom signals** | No (needs human labels) | Via reward model | Yes (any Python function) |
| **Used by** | Most open-source models | ChatGPT, Claude | DeepSeek-R1 |

---

## The SFT → Alignment Pipeline

```
Stage 1: SFT ─────────────────────────────────────
  Input:  Base model + instruction data
  Output: Model that follows instructions
  Steps:  1-7 in trainer.py

Stage 2: Alignment (pick one) ─────────────────────
  DPO:   Feed preference pairs → update model
  PPO:   Generate → score with reward model → PPO update
  GRPO:  Generate N responses → score with function → group update

  Output: Model that prefers high-quality responses

Stage 3: Export ───────────────────────────────────
  Merge adapters → quantize → push to Hub
```

---

## Hyperparameter Cheat Sheet

| Parameter | DPO | PPO | GRPO |
|-----------|-----|-----|------|
| Learning rate | 5e-5 | 1e-5 | 5e-5 |
| Epochs | 1 | N/A (step-based) | 1 |
| Batch size | 2 | 2 | 2 |
| Key param | β = 0.1 | ppo_epochs = 4 | num_generations = 4 |

**Key rule**: Alignment learning rate should be **lower** than SFT (typically 5-10× lower).

---

## Tips

1. **Always SFT first** — alignment assumes the model already knows how to generate
2. **DPO for most use cases** — simplest, most stable, best documented
3. **PPO for maximum quality** — if you have the compute and a good reward model
4. **GRPO for custom signals** — when you want programmatic control over what "good" means
5. **1 epoch is usually enough** — more can degrade performance (over-alignment)
6. **Lower LR** — 5e-5 or lower to prevent catastrophic changes

---

## Next: [Architecture Deep-Dive →](09-architecture-deep-dive.md)
