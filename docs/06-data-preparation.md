# 6. Data Preparation

> **Goal**: Understand data formats (Alpaca, ShareGPT, tool-calling), how tokenization works, and how lfm-trainer cleans and prepares your data.

---

## Data Formats

Language models learn from examples. The format of those examples matters.

### 1. Alpaca Format (instruction-response)

The simplest format. Each example has an instruction and a response:

```json
{
  "instruction": "Write a Python function to reverse a string",
  "input": "",
  "output": "def reverse_string(s):\n    return s[::-1]"
}
```

With optional input context:

```json
{
  "instruction": "Translate the following to French",
  "input": "Hello, how are you?",
  "output": "Bonjour, comment allez-vous?"
}
```

**When to use**: Simple tasks, Q&A, single-turn interactions.

### 2. Conversational / ShareGPT Format

Multi-turn conversations with roles:

```json
{
  "conversations": [
    {"role": "system", "content": "You are a coding assistant."},
    {"role": "user", "content": "How do I sort a list in Python?"},
    {"role": "assistant", "content": "You can use sorted(): sorted([3,1,2])"},
    {"role": "user", "content": "What about in-place sorting?"},
    {"role": "assistant", "content": "Use .sort(): my_list.sort()"}
  ]
}
```

**When to use**: Chatbots, multi-turn dialogues.

### 3. Tool-Calling Format

Conversations where the assistant uses tools/functions:

```json
{
  "conversations": [
    {"role": "user", "content": "Click the login button"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"function": {"name": "click", "arguments": "{\"selector\": \"#login\"}"}}
    ]},
    {"role": "tool", "content": "Button clicked successfully"},
    {"role": "assistant", "content": "I clicked the login button for you."}
  ]
}
```

**When to use**: Training models to use APIs, MCP tools, function calling.

### 4. Plain Text

Just raw text. Used for continued pre-training rather than instruction tuning:

```json
{"text": "def fibonacci(n):\n    if n <= 1: return n\n    return fibonacci(n-1) + fibonacci(n-2)"}
```

**When to use**: Domain adaptation (e.g., teaching a model about a new codebase).

---

## How Tokenization Works Behind the Scenes

### Chat Templates

Modern models use **chat templates** to format conversations into a single token sequence:

```
Original conversation:
  User: "Hello"
  Assistant: "Hi there!"

Tokenized with template:
  <|im_start|>user
  Hello<|im_end|>
  <|im_start|>assistant
  Hi there!<|im_end|>
```

The special tokens (`<|im_start|>`, `<|im_end|>`) tell the model where each turn begins and ends.

### Loss Masking

**Critical concept**: We only compute loss on the **assistant's responses**, not the user's messages.

```
<|im_start|>user
How do I sort a list?<|im_end|>     ← loss = 0 (masked)
<|im_start|>assistant
Use sorted().<|im_end|>             ← loss computed here!
```

**Why?** We want the model to learn generating good responses, not memorizing user messages.

### Padding and Truncation

Batches need uniform length. We handle this with:

```
Example 1: [token₁, token₂, token₃, token₄, token₅]           (5 tokens)
Example 2: [token₁, token₂, token₃]                           (3 tokens)

After padding (max_seq_length=5):
Example 1: [token₁, token₂, token₃, token₄, token₅]           (unchanged)
Example 2: [token₁, token₂, token₃, PAD,    PAD   ]           (padded)

Attention mask:
Example 1: [1, 1, 1, 1, 1]  ← attend to all
Example 2: [1, 1, 1, 0, 0]  ← ignore padding
```

If sequences exceed `max_seq_length` (default: 2048), they're truncated.

---

## Data Quality Filters

Bad data → bad model. lfm-trainer provides automatic cleaning:

### 1. Empty row removal

```python
# Before: dataset has empty/whitespace rows
{"instruction": "", "output": ""}        ← removed
{"instruction": "   ", "output": "  "}   ← removed
{"instruction": "Sort a list", "output": "sorted(lst)"}  ← kept
```

### 2. Length filtering

```python
cfg = TrainingConfig(
    quality_filter=True,
    min_length=10,      # Remove examples shorter than 10 chars
    max_length=10000,   # Remove examples longer than 10000 chars
)
```

**Why?** Very short examples don't teach much. Very long examples waste compute and hit truncation.

### 3. Deduplication

Exact duplicates are removed using text hashing:

```python
# These are duplicates — only one is kept:
{"text": "def hello(): print('hi')"}
{"text": "def hello(): print('hi')"}  ← removed
```

### Behind the scenes (the math)

Deduplication uses a hash set:

```python
seen_hashes = set()
for example in dataset:
    h = hash(example["text"])
    if h in seen_hashes:
        continue  # skip duplicate
    seen_hashes.add(h)
    clean_data.append(example)
```

---

## Dataset Sources

lfm-trainer loads data from multiple sources automatically:

| Source | Example | How it works |
|--------|---------|-------------|
| HuggingFace Hub | `"sahil2801/CodeAlpaca-20k"` | Auto-downloaded via `datasets` library |
| Local CSV | `"./data/my_data.csv"` | Loaded with pandas |
| Local JSON/JSONL | `"./data/prompts.jsonl"` | One JSON object per line |
| Local Parquet | `"./data/training.parquet"` | Columnar format, fast |

### Combining multiple datasets

```python
cfg = TrainingConfig(
    dataset_paths=[
        "sahil2801/CodeAlpaca-20k",           # HF Hub
        "./my_custom_data.jsonl",              # Local
        "jdaddyalbs/playwright-mcp-toolcalling", # HF Hub
    ],
)
# All datasets are loaded, formatted, and concatenated
```

---

## Evaluation Split

Holding out data for validation helps detect overfitting:

```python
cfg = TrainingConfig(eval_split=0.1)  # 10% for evaluation
```

```
Dataset: 10,000 examples
├── Train: 9,000 (90%) — model learns from these
└── Eval:  1,000 (10%) — model is tested on these (never trained on)

During training:
  Step 100: train_loss=2.1, eval_loss=2.3  ← both going down, good
  Step 200: train_loss=1.5, eval_loss=1.7  ← still good
  Step 300: train_loss=0.8, eval_loss=1.2  ← gap widening... 
  Step 400: train_loss=0.3, eval_loss=1.5  ← overfitting! eval is rising
```

---

## Tool-Calling Data Filter

When training for tool use, you may want only examples that contain tool calls:

```python
cfg = TrainingConfig(tool_calling_only=True)
```

This filters out regular conversation examples and keeps only those with `tool_calls` in the assistant messages.

---

## The Complete Data Pipeline

```
Raw dataset               Quality filters           Formatting            Tokenization
┌──────────┐    ┌────────────────────┐    ┌──────────────────┐    ┌─────────────┐
│ CSV      │───→│ Remove empty rows  │───→│ Detect format    │───→│ Apply chat  │
│ JSON     │    │ Length filter       │    │ (Alpaca/ShareGPT │    │ template    │
│ Parquet  │    │ Deduplicate        │    │  /text/tool)     │    │ Tokenize    │
│ HF Hub   │    │ Tool-calling filter│    │ Normalize to     │    │ Pad/Truncate│
└──────────┘    └────────────────────┘    │ conversation fmt │    │ Create masks│
                                          └──────────────────┘    └─────────────┘
                                                                         │
                                                                         ▼
                                                                  Train / Eval
                                                                     Split
```

---

## Next: [Evaluation & Benchmarks →](07-evaluation-and-benchmarks.md)
