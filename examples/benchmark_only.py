"""
Example: Benchmark-only — evaluate an existing model without training.

Useful for:
  - Evaluating a model you already trained
  - Comparing two checkpoints
  - Getting baseline scores before deciding to fine-tune

Usage:
    python examples/benchmark_only.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lfm_trainer.benchmark import run_benchmarks, format_benchmark_report

# Load any model (base, fine-tuned, or merged)
MODEL_ID = "liquid/LFM2.5-1.2B-Base"

print(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Run selected benchmarks
results = run_benchmarks(
    model=model,
    tokenizer=tokenizer,
    benchmarks=["humaneval", "mbpp"],  # or ["all"] for everything
    max_problems=20,                   # Remove for full eval
)

# Print results
print("\n" + format_benchmark_report(results))

# Results are also available programmatically
for r in results:
    print(f"{r.benchmark}: pass@1 = {r.pass_at_1 * 100:.1f}% ({r.num_correct}/{r.num_problems})")
