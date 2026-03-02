"""
Auto-benchmarking on popular coding benchmarks (HumanEval, MBPP).

Runs evaluation before and after training to measure improvement,
and optionally publishes results to the HuggingFace model card.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Holds results from a single benchmark run."""

    benchmark: str
    pass_at_1: float = 0.0
    pass_at_5: float = 0.0
    pass_at_10: float = 0.0
    num_problems: int = 0
    num_correct: int = 0
    raw_results: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "pass@1": round(self.pass_at_1 * 100, 2),
            "pass@5": round(self.pass_at_5 * 100, 2),
            "pass@10": round(self.pass_at_10 * 100, 2),
            "num_problems": self.num_problems,
            "num_correct": self.num_correct,
        }


@dataclass
class BenchmarkComparison:
    """Before/after comparison of benchmark results."""

    before: BenchmarkResult
    after: BenchmarkResult

    @property
    def delta_pass_at_1(self) -> float:
        return self.after.pass_at_1 - self.before.pass_at_1

    def summary_table(self) -> str:
        """Generate a markdown table comparing before/after results."""
        b, a = self.before.to_dict(), self.after.to_dict()
        delta_1 = self.delta_pass_at_1 * 100
        sign = "+" if delta_1 >= 0 else ""

        return (
            f"| Metric | Before | After | Delta |\n"
            f"|--------|--------|-------|-------|\n"
            f"| pass@1 | {b['pass@1']:.1f}% | {a['pass@1']:.1f}% | {sign}{delta_1:.1f}% |\n"
            f"| pass@5 | {b['pass@5']:.1f}% | {a['pass@5']:.1f}% | — |\n"
            f"| pass@10 | {b['pass@10']:.1f}% | {a['pass@10']:.1f}% | — |\n"
        )


# ── HumanEval ─────────────────────────────────────────────────────────────

def _generate_completion(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    n_samples: int = 1,
) -> list[str]:
    """Generate code completions for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    completions = []

    for _ in range(n_samples):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        # Decode only the generated part
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True)

        # Stop at common end-of-function markers
        for stop in ["\nclass ", "\ndef ", "\n#", "\nif __name__", "\nprint("]:
            if stop in text:
                text = text[: text.index(stop)]
                break

        completions.append(text)

    return completions


def _run_humaneval(
    model,
    tokenizer,
    n_samples: int = 1,
    max_problems: Optional[int] = None,
) -> BenchmarkResult:
    """Run HumanEval benchmark."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library required for benchmarking")
        return BenchmarkResult(benchmark="humaneval")

    logger.info("📊 Running HumanEval benchmark (n=%d)…", n_samples)
    ds = load_dataset("openai/openai_humaneval", split="test")

    if max_problems:
        ds = ds.select(range(min(max_problems, len(ds))))

    correct = 0
    total = len(ds)
    raw_results = []

    for idx, problem in enumerate(ds):
        prompt = problem["prompt"]
        test_code = problem["test"]
        entry_point = problem["entry_point"]

        completions = _generate_completion(model, tokenizer, prompt, n_samples=n_samples)

        problem_pass = False
        for completion in completions:
            full_code = prompt + completion + "\n" + test_code + f"\ncheck({entry_point})\n"
            try:
                exec(full_code, {"__builtins__": __builtins__}, {})
                problem_pass = True
                break
            except Exception:
                pass

        if problem_pass:
            correct += 1

        raw_results.append({
            "task_id": problem["task_id"],
            "passed": problem_pass,
            "completion": completions[0][:200],
        })

        if (idx + 1) % 20 == 0:
            logger.info("  HumanEval progress: %d/%d (%.1f%% so far)", idx + 1, total, 100 * correct / (idx + 1))

    pass_at_1 = correct / total if total > 0 else 0.0

    logger.info("✅ HumanEval: pass@1 = %.1f%% (%d/%d)", pass_at_1 * 100, correct, total)

    return BenchmarkResult(
        benchmark="humaneval",
        pass_at_1=pass_at_1,
        num_problems=total,
        num_correct=correct,
        raw_results=raw_results,
    )


def _run_mbpp(
    model,
    tokenizer,
    n_samples: int = 1,
    max_problems: Optional[int] = None,
) -> BenchmarkResult:
    """Run MBPP (Mostly Basic Python Problems) benchmark."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets library required for benchmarking")
        return BenchmarkResult(benchmark="mbpp")

    logger.info("📊 Running MBPP benchmark (n=%d)…", n_samples)
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    if max_problems:
        ds = ds.select(range(min(max_problems, len(ds))))

    correct = 0
    total = len(ds)
    raw_results = []

    for idx, problem in enumerate(ds):
        prompt = f"# {problem['prompt']}\n\ndef solution():\n"
        test_cases = problem["test_list"]

        completions = _generate_completion(model, tokenizer, prompt, n_samples=n_samples)

        problem_pass = False
        for completion in completions:
            full_code = prompt + completion + "\n" + "\n".join(test_cases)
            try:
                exec(full_code, {"__builtins__": __builtins__}, {})
                problem_pass = True
                break
            except Exception:
                pass

        if problem_pass:
            correct += 1

        raw_results.append({
            "task_id": problem["task_id"],
            "passed": problem_pass,
        })

        if (idx + 1) % 20 == 0:
            logger.info("  MBPP progress: %d/%d (%.1f%% so far)", idx + 1, total, 100 * correct / (idx + 1))

    pass_at_1 = correct / total if total > 0 else 0.0
    logger.info("✅ MBPP: pass@1 = %.1f%% (%d/%d)", pass_at_1 * 100, correct, total)

    return BenchmarkResult(
        benchmark="mbpp",
        pass_at_1=pass_at_1,
        num_problems=total,
        num_correct=correct,
        raw_results=raw_results,
    )


# ── Public API ────────────────────────────────────────────────────────────

def run_benchmarks(
    model,
    tokenizer,
    benchmarks: list[str] | None = None,
    n_samples: int = 1,
    max_problems: Optional[int] = None,
) -> list[BenchmarkResult]:
    """Run coding benchmarks on a model.

    Parameters
    ----------
    model:
        The model to evaluate (can be PeftModel or base).
    tokenizer:
        The tokenizer.
    benchmarks:
        List of benchmark names. Default: ``["humaneval", "mbpp"]``.
    n_samples:
        Number of completions to generate per problem.
    max_problems:
        Cap the number of problems (for quick testing).

    Returns
    -------
    List of BenchmarkResult objects.
    """
    if benchmarks is None:
        benchmarks = ["humaneval", "mbpp"]

    model.eval()
    results = []

    for bench in benchmarks:
        if bench == "humaneval":
            results.append(_run_humaneval(model, tokenizer, n_samples, max_problems))
        elif bench == "mbpp":
            results.append(_run_mbpp(model, tokenizer, n_samples, max_problems))
        else:
            logger.warning("Unknown benchmark: %s — skipping", bench)

    return results


def run_before_after_benchmark(
    model_before,
    model_after,
    tokenizer,
    benchmarks: list[str] | None = None,
    n_samples: int = 1,
    max_problems: Optional[int] = None,
) -> list[BenchmarkComparison]:
    """Run benchmarks on both the base and fine-tuned model, return comparisons."""
    logger.info("═══ Running BEFORE benchmarks (base model) ═══")
    before_results = run_benchmarks(model_before, tokenizer, benchmarks, n_samples, max_problems)

    logger.info("═══ Running AFTER benchmarks (fine-tuned) ═══")
    after_results = run_benchmarks(model_after, tokenizer, benchmarks, n_samples, max_problems)

    comparisons = []
    for b, a in zip(before_results, after_results):
        comp = BenchmarkComparison(before=b, after=a)
        logger.info(
            "📈 %s: pass@1 %.1f%% → %.1f%% (%+.1f%%)",
            b.benchmark,
            b.pass_at_1 * 100,
            a.pass_at_1 * 100,
            comp.delta_pass_at_1 * 100,
        )
        comparisons.append(comp)

    return comparisons


def format_benchmark_report(
    results: list[BenchmarkResult] | list[BenchmarkComparison],
) -> str:
    """Format benchmark results as a markdown report for model cards."""
    lines = ["## Benchmark Results\n"]

    for item in results:
        if isinstance(item, BenchmarkComparison):
            lines.append(f"### {item.before.benchmark}\n")
            lines.append(item.summary_table())
        else:
            r = item.to_dict()
            lines.append(f"### {r['benchmark']}\n")
            lines.append(f"| Metric | Score |")
            lines.append(f"|--------|-------|")
            lines.append(f"| pass@1 | {r['pass@1']:.1f}% |")
            lines.append(f"| Problems | {r['num_problems']} |")
            lines.append(f"| Correct | {r['num_correct']} |")
        lines.append("")

    return "\n".join(lines)
