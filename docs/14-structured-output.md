# 14. Structured Output (JSON Mode)

> **Goal**: Train models to generate valid JSON conforming to schemas — critical for reliable tool calling, API integrations, and data extraction.

---

## Why Structured Output?

Without structured output training, models produce unreliable JSON:

```
User: "Extract name and age from: John is 30 years old"
Model: The name is John and he is 30 years old.    ← Not JSON!
Model: {"name": "John", age: 30}                   ← Invalid JSON!
Model: {"name": "John"}                             ← Missing "age"!
```

With structured output training:

```
User: "Extract name and age from: John is 30 years old"
Model: {"name": "John", "age": 30}                 ← Valid ✓ Complete ✓
```

---

## How It Works

### 1. Schema-augmented training

Each training example includes the JSON schema in the prompt:

```
### System:
You are a helpful assistant that responds ONLY with valid JSON.
Your response must conform to the following JSON schema:
```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"}
  },
  "required": ["name", "age"]
}
```
Respond with ONLY the JSON object, no additional text.

### User:
Extract info: John Doe, age 30

### Assistant:
{"name": "John Doe", "age": 30}
```

### 2. Built-in schemas (6 types)

| Schema | Use Case |
|--------|----------|
| **person** | Contact info extraction |
| **api_response** | HTTP API formatting |
| **task_list** | Task management |
| **function_call** | Tool calling (structured) |
| **code_review** | Code feedback |
| **search_results** | Search response formatting |

### 3. Validation pipeline

The module includes a lightweight JSON Schema validator (no external dependency):

```python
from lfm_trainer.structured_output import validate_json, validate_against_schema

# Check valid JSON
is_valid, obj, err = validate_json('{"name": "Alice"}')

# Check schema conformance
schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
is_valid, errors = validate_against_schema('{"name": "Alice"}', schema)
```

---

## Usage

### CLI

```bash
lfm-train --dataset sahil2801/CodeAlpaca-20k \
    --structured-output \
    --benchmarks json_output toolcall \
    --hub-repo your-username/lfm-json-master
```

### Python API

```python
from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training

cfg = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    structured_output=True,          # Mix in JSON schema training data
    benchmark_names=["json_output"],  # Test JSON generation quality
)
run_training(cfg)
```

### Custom schemas

```python
from lfm_trainer.structured_output import create_structured_output_dataset

custom_schemas = [
    {
        "name": "deployment",
        "description": "Deployment config",
        "schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "version": {"type": "string"},
                "replicas": {"type": "integer", "minimum": 1},
                "environment": {"type": "string", "enum": ["dev", "staging", "production"]},
            },
            "required": ["service", "version", "environment"],
        },
    },
]

dataset = create_structured_output_dataset(schemas=custom_schemas, samples_per_schema=50)
```

---

## The `json_output` Benchmark

Tests the model's ability to generate schema-conformant JSON:

| Metric | What it measures |
|--------|-----------------|
| **Schema validity** | Does the output match all required fields, types, enums? |
| **JSON validity** | Is the output parseable JSON? |
| **Pass@1** | Schema validity rate across all test prompts |

```bash
lfm-train --benchmarks json_output
```

Total benchmarks now: **9** (humaneval, mbpp, multiple, bigcodebench, evalplus, toolcall, gsm8k, reasoning, json_output)

---

## Tips

1. **Combine with tool calling** — `--structured-output` + `--tool-calling-only` for maximum reliability
2. **Custom schemas for your domain** — pass your own schemas to `create_structured_output_dataset`
3. **Use at inference time** — import `validate_json` and `validate_against_schema` to check model outputs
4. **Lower temperature** — use temperature 0.1–0.3 at inference for more deterministic JSON

---

## Example

See: [`examples/structured_output.py`](../examples/structured_output.py)

## Next: Back to [README →](../README.md)
