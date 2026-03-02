# 7. Evaluation & Benchmarks

> **Goal**: Understand how coding benchmarks work, what pass@k means, and how lfm-trainer evaluates your model.

---

## Why Benchmark?

Loss tells you how well the model predicts tokens, but not whether it can **solve real problems**.

```
Model A: loss = 1.2  → Can it actually write working code? 🤷
Model B: loss = 1.5  → Maybe it writes better code despite higher loss?
```

Benchmarks test the model on actual programming tasks with **executable test cases**.

---

## How a Benchmark Works

Every benchmark follows the same pattern:

```
1. Give the model a problem description (prompt)
2. Model generates a solution (code)
3. Run the solution against test cases
4. Record: did it pass? ✅ or ❌
5. Repeat for all problems
6. Report: what % of problems passed?
```

---

## The pass@k Metric

The most important metric for coding benchmarks.

### pass@1

"Generate **one** solution per problem. What fraction pass all tests?"

```
Problem 1: ✅ passed
Problem 2: ❌ failed
Problem 3: ✅ passed
Problem 4: ❌ failed
Problem 5: ✅ passed

pass@1 = 3/5 = 60%
```

### pass@k (k > 1)

"Generate **k** solutions per problem. Does **at least one** pass?"

```
Problem 1, attempt 1: ❌
Problem 1, attempt 2: ✅  → Problem 1: PASSED (at least one worked)

Problem 2, attempt 1: ❌
Problem 2, attempt 2: ❌  → Problem 2: FAILED (none worked)
```

### The math behind pass@k

Given `n` total samples and `c` correct ones:

```
pass@k = 1 - C(n-c, k) / C(n, k)
```

Where `C(a, b)` is "a choose b" (combinations).

**Intuition**: pass@k = 1 - (probability that all k samples are wrong)

For practical purposes:
- **pass@1**: strictest — model must get it right in one shot
- **pass@5**: model gets 5 tries
- **pass@10**: model gets 10 tries

Higher k → higher scores (more chances to get lucky).

---

## Benchmarks in lfm-trainer

### 1. HumanEval (OpenAI)

| Detail | Value |
|--------|-------|
| Problems | 164 |
| Language | Python |
| Difficulty | Medium |
| What it tests | Function completion from docstring |

**Example problem:**

```python
# Prompt given to the model:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers 
    closer to each other than given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

```python
# Model generates:
    for i, n1 in enumerate(numbers):
        for j, n2 in enumerate(numbers):
            if i != j and abs(n1 - n2) < threshold:
                return True
    return False
```

```python
# Test cases verify:
assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False  ✅
assert has_close_elements([1.0, 2.8, 3.0], 0.3) == True   ✅
```

### 2. MBPP (Google)

| Detail | Value |
|--------|-------|
| Problems | 427 (sanitized split) |
| Language | Python |
| Difficulty | Beginner |
| What it tests | Simple algorithmic problems |

**Example:**

```python
# Prompt: Write a function to find the minimum cost path in a grid.
# Test: assert min_cost([[1,2,3],[4,5,6]]) == 12
```

### 3. MultiPL-E

| Detail | Value |
|--------|-------|
| Problems | 164 × 7 languages |
| Languages | Python, JavaScript, TypeScript, Java, C++, Rust, Go |
| What it tests | Cross-language generation ability |

Takes HumanEval problems and translates them. Tests if your model can generate code in multiple languages, not just Python.

### 4. BigCodeBench (ICLR 2025)

| Detail | Value |
|--------|-------|
| Problems | 1,140 |
| Language | Python |
| Difficulty | Hard |
| What it tests | Multi-library, real-world tasks |

Problems require using libraries like `pandas`, `numpy`, `matplotlib`, `requests`. Much harder than HumanEval.

### 5. EvalPlus / HumanEval+

| Detail | Value |
|--------|-------|
| Problems | 164 (same as HumanEval) |
| Extra tests | 80× more test cases per problem |
| What it tests | Edge cases and robustness |

A model might pass HumanEval's 7 tests but fail EvalPlus's 560 tests for the same problem. EvalPlus catches solutions that work on basic cases but fail on edge cases.

---

## Before/After Comparison

The most useful feature: benchmark BOTH the base model and your fine-tuned model.

```
Base model (before training):
  HumanEval pass@1: 12.2%
  MBPP pass@1:      18.5%

Fine-tuned model (after training):
  HumanEval pass@1: 28.4%  (+16.2%)
  MBPP pass@1:      34.1%  (+15.6%)
```

This **quantifies** your fine-tuning improvement.

### How it works in lfm-trainer

```python
cfg = TrainingConfig(
    run_benchmark=True,
    benchmark_before_after=True,  # runs on base model too
    benchmark_max_problems=20,    # quick test with 20 problems
)
```

Behind the scenes:
1. Load a fresh copy of the base model
2. Run benchmarks on base model → "before" scores
3. Run benchmarks on your fine-tuned model → "after" scores
4. Compute deltas and generate comparison table

---

## Safe Code Execution

Running generated code is **dangerous** — it could delete files, run infinite loops, or worse.

lfm-trainer uses **subprocess isolation** with timeouts:

```python
# Each solution runs in a separate process with a 5-second timeout
result = subprocess.run(
    [sys.executable, "-c", generated_code],
    capture_output=True,
    timeout=5.0,  # kill after 5 seconds
)
```

The generated code:
- Runs in a **separate process** (can't access your Python state)
- Has a **timeout** (5 seconds default, 10 for BigCodeBench)
- Only checks return code (pass/fail)

---

## Typical Scores for 1.2B Models

| Benchmark | Base model | After LoRA fine-tune | After full fine-tune |
|-----------|-----------|---------------------|---------------------|
| HumanEval | 5-15% | 15-30% | 20-35% |
| MBPP | 10-20% | 20-40% | 25-45% |
| BigCodeBench | 1-5% | 5-15% | 8-20% |

**Note**: 1.2B models are small! Don't expect GPT-4 level performance (50-90%). Even a 10% improvement is significant at this scale.

---

## Next: [Quantization & Export →](08-quantization-and-export.md)
