# 📚 LFM Trainer — Documentation

Beginner-friendly guides explaining LLM fine-tuning from the ground up.  
No prerequisites beyond basic Python and high-school math.

## Guides

| # | Guide | What you'll learn |
|---|-------|-------------------|
| 1 | [What is an LLM?](01-what-is-an-llm.md) | Tokens, embeddings, transformers, attention — the building blocks |
| 2 | [How Training Works](02-how-training-works.md) | Loss functions, backpropagation, gradient descent — the math |
| 3 | [Fine-Tuning vs Training from Scratch](03-fine-tuning-explained.md) | Why we don't train from zero, transfer learning, catastrophic forgetting |
| 4 | [LoRA & Parameter-Efficient Training](04-lora-explained.md) | Low-rank adapters, why they work, the math behind LoRA |
| 5 | [Full Fine-Tuning](05-full-fine-tuning.md) | When to use it, memory requirements, gradient checkpointing |
| 6 | [Data Preparation](06-data-preparation.md) | Formats (Alpaca, ShareGPT, tool-calling), cleaning, tokenization |
| 7 | [Evaluation & Benchmarks](07-evaluation-and-benchmarks.md) | HumanEval, MBPP, pass@k metric, how benchmarks actually work |
| 8 | [Quantization & Export](08-quantization-and-export.md) | GGUF, MLX, INT4/INT8, why smaller models run faster |
| 9 | [Architecture Deep-Dive](09-architecture-deep-dive.md) | How lfm-trainer is built — every module explained |
| 10 | [DPO, PPO, GRPO & Alignment](10-dpo-and-alignment.md) | DPO/PPO/GRPO math, datasets, SFT→alignment pipeline |
| 11 | [Continued Pre-Training (CPT)](11-continued-pretraining.md) | Train on books, raw text, inject domain knowledge |

## Quick Start

If you just want to train a model, see the [main README](../README.md).  
These docs are for understanding **why** things work, not just **how** to use them.

## Prerequisites

- Basic Python (functions, classes, loops)
- High-school math (matrices, derivatives — we'll refresh everything)
- Curiosity 🧠
