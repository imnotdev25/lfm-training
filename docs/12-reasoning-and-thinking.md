# 12. Reasoning, Thinking, and Tool Calling

> **Goal**: Understand how to train a model that THINKS before acting, using `<think>` tags for internal reasoning alongside tool calling.

---

## Why Reasoning Matters

Without reasoning, a model jumps straight to an answer:

```
User: "What's 25% of the items that cost more than $50?"
Model: "$12.50"  ← wrong, didn't think it through
```

With `<think>` tags, the model reasons step-by-step first:

```
User: "What's 25% of the items that cost more than $50?"
Model:
<think>
Let me break this down:
1. First, I need to find items costing more than $50
2. Then calculate 25% of those items
3. Looking at the data: 8 items total, 4 cost > $50
4. 25% of 4 = 1 item
</think>
There is 1 item that represents 25% of those costing more than $50.
```

---

## How `<think>` Tags Work

The `<think>...</think>` tags are placed **inside the assistant's response**, before the actual content:

```
### Assistant:
<think>
Internal reasoning — the model's "scratch pad"
This is where it plans, analyzes, and decides
</think>

The actual response to the user goes here.
```

### With tool calling

When the model needs to call a tool, it thinks first, then acts:

```
### Assistant:
<think>
The user wants weather in San Francisco.
I should use the get_weather function with location="San Francisco" and unit="celsius".
</think>

<|tool_call_start|>
get_weather(location="San Francisco", unit="celsius")
<|tool_call_end|>
```

### The key pattern: Think → Content → Tool Call

```
┌─────────────────────────────────────┐
│ ### Assistant:                       │
│                                      │
│ <think>                              │  ← Step 1: Reason
│   Internal reasoning...              │
│ </think>                             │
│                                      │
│ Here's what I found...               │  ← Step 2: Respond
│                                      │
│ <|tool_call_start|>                  │  ← Step 3: Act
│   function_name(args)                │
│ <|tool_call_end|>                    │
└─────────────────────────────────────┘
```

---

## The TxT360 Dataset

**[LLM360/TxT360-3efforts](https://huggingface.co/datasets/LLM360/TxT360-3efforts)** is perfect for training reasoning + tool calling:

| Feature | Description |
|---------|-----------|
| Format | Multi-turn conversations |
| Size | ~1M+ rows (we use first 100K) |
| Think field | `think` key in assistant messages |
| Tool calls | Full tool definitions + function calls |
| Tool results | Tool execution results in conversation |

### Dataset structure

```json
{
  "messages": [
    {
      "role": "system",
      "content": "",
      "tools": [
        {"name": "search_web", "parameters": {...}},
        {"name": "get_weather", "parameters": {...}}
      ]
    },
    {
      "role": "user",
      "content": "Find me flights from NYC to London"
    },
    {
      "role": "assistant",
      "think": "The user wants flight information. I should search for flights...",
      "content": "I'll search for flights for you.",
      "tool_calls": [{"function": {"name": "search_flights", "arguments": "..."}}]
    }
  ]
}
```

The `think` field is automatically captured by lfm-trainer and wrapped in `<think>...</think>` tags.

---

## Training for Reasoning

### Python API

```python
from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["LLM360/TxT360-3efforts"],
    enable_reasoning=True,
    reasoning_dataset="LLM360/TxT360-3efforts",
    reasoning_max_samples=100_000,      # First 100K rows
    max_seq_length=4096,                # Reasoning traces need more context
    hub_repo_id="your-username/lfm-reasoner",
)
run_training(cfg)
```

### CLI

```bash
lfm-train --dataset LLM360/TxT360-3efforts \
    --enable-reasoning \
    --reasoning-dataset LLM360/TxT360-3efforts \
    --reasoning-max-samples 100000 \
    --benchmarks toolcall gsm8k reasoning humaneval
```

---

## Benchmarks

### New benchmarks for reasoning and tool calling

| Benchmark | What it tests | Problems | Dataset |
|-----------|--------------|----------|---------|
| **toolcall** | Function name + argument extraction | ~55 | Salesforce/xlam + built-in |
| **gsm8k** | Math word problem reasoning | 1,319 | openai/gsm8k |
| **reasoning** | Scientific reasoning (ARC-Challenge) | 1,172 | allenai/ai2_arc |

### Existing coding benchmarks

| Benchmark | What it tests | Problems |
|-----------|--------------|----------|
| humaneval | Python code generation | 164 |
| mbpp | Python code generation | 427 |
| multiple | Multi-language code | ~1,400 |
| bigcodebench | Complex functions | 1,140 |
| evalplus | HumanEval+ (more tests) | 164 |

### Using benchmarks

```bash
# Tool calling only
lfm-train --benchmarks toolcall

# Reasoning suite
lfm-train --benchmarks gsm8k reasoning

# Full suite
lfm-train --benchmarks humaneval mbpp toolcall gsm8k reasoning

# All available
lfm-train --benchmarks all
```

---

## Three Model Recipes

### Recipe 1: Tool Calling Specialist

```
Goal: AI agent that calls APIs correctly
Data: Salesforce/xlam-function-calling-60k
Pipeline: SFT → toolcall benchmark
```

```python
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["Salesforce/xlam-function-calling-60k"],
    tool_calling_only=True,
    benchmark_names=["toolcall", "humaneval"],
)
```

See: [`examples/recipe_tool_calling.py`](../examples/recipe_tool_calling.py)

---

### Recipe 2: Reasoning + Tool Calling

```
Goal: Model that thinks before acting
Data: TxT360 (100K) + CodeAlpaca
Pipeline: SFT with <think> tags → DPO → full benchmark
```

```python
cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k", "LLM360/TxT360-3efforts"],
    enable_reasoning=True,
    reasoning_dataset="LLM360/TxT360-3efforts",
    reasoning_max_samples=100_000,
    alignment_dataset="argilla/dpo-mix-7k",
    benchmark_names=["toolcall", "gsm8k", "reasoning", "humaneval"],
)
```

See: [`examples/recipe_reasoning_tools.py`](../examples/recipe_reasoning_tools.py)

---

### Recipe 3: Domain Expert from Scratch

```
Goal: Specialized expert (coding, system design, engineering)
Data: Books + blogs → instructions → preferences
Pipeline: CPT → SFT → DPO
```

```python
# Stage 1: CPT on books and blogs
cfg_cpt = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    cpt_sources=["/path/to/books/", "/path/to/blogs/"],
    cpt_epochs=2,
)

# Stage 2: SFT on instructions  
cfg_sft = TrainingConfig(
    resume_from_model="your-username/lfm-domain-base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
)

# Stage 3: DPO alignment
cfg_dpo = TrainingConfig(
    resume_from_model="your-username/lfm-domain-sft",
    alignment_dataset="argilla/dpo-mix-7k",
)
```

See: [`examples/recipe_from_scratch.py`](../examples/recipe_from_scratch.py)

---

## Recommended System Design Resources

### Open-Source Repositories

| Resource | Stars | Topics |
|----------|-------|--------|
| [donnemartin/system-design-primer](https://github.com/donnemartin/system-design-primer) | 280K+ | Complete system design guide |
| [karanpratapsingh/system-design](https://github.com/karanpratapsingh/system-design) | 35K+ | IP, DNS, load balancing, databases |
| [ashishps1/awesome-system-design-resources](https://github.com/ashishps1/awesome-system-design-resources) | 20K+ | Curated links and guides |
| [binhnguyennus/awesome-scalability](https://github.com/binhnguyennus/awesome-scalability) | 58K+ | Scalability patterns |
| [ByteByteGoHq/system-design-101](https://github.com/ByteByteGoHq/system-design-101) | 65K+ | Visual system design |

### Engineering Blogs (for CPT)

| Company | Blog URL | Focus |
|---------|----------|-------|
| Netflix | netflixtechblog.com | Microservices, streaming |
| Uber | uber.com/blog/engineering | Distributed systems |
| Meta | engineering.fb.com | Scale, ML infra |
| Google AI | ai.googleblog.com | ML, research |
| Stripe | stripe.com/blog/engineering | Payments, API design |
| Airbnb | medium.com/airbnb-engineering | Search, recommendations |
| Shopify | shopify.engineering | E-commerce systems |
| LinkedIn | engineering.linkedin.com | Data pipelines |
| Cloudflare | blog.cloudflare.com | CDN, security, edge |
| AWS | aws.amazon.com/blogs/architecture | Cloud architecture |

### Coding Textbooks (free / open-source)

| Book | Author | License |
|------|--------|---------|
| Think Python | Allen B. Downey | CC |
| Automate the Boring Stuff | Al Sweigart | CC |
| Eloquent JavaScript | Marijn Haverbeke | CC |
| The Rust Book | Steve Klabnik | MIT |
| Go by Example | Mark McGranaghan | CC |

### HuggingFace Datasets (raw text)

| Dataset | Type | Size |
|---------|------|------|
| HuggingFaceTB/cosmopedia | Synthetic textbooks | Large |
| wikimedia/wikipedia | General knowledge | 6M+ articles |
| bigcode/the-stack-v2-train-smol-ids | Source code | Large |

---

## Tips

1. **Longer context for reasoning** — set `max_seq_length=4096` or higher since `<think>` blocks add length
2. **First 100K rows of TxT360** — already high quality, no need to use the full dataset initially
3. **Benchmark with toolcall + gsm8k** — these directly measure reasoning and tool use quality
4. **DPO after reasoning SFT** — helps the model produce cleaner, more helpful final answers
5. **Test your `<think>` output** — check that the model actually uses the tags at inference time

---

## Next: Back to [README →](../README.md)
