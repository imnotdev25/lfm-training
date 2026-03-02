"""
Structured Output Training — teach models to generate valid JSON.

This example trains the model to output valid JSON conforming to schemas.
Critical for reliable tool calling, API integrations, and structured data extraction.

Includes 6 built-in schemas:
  - person: Extract person information
  - api_response: Format API responses
  - task_list: Create structured task lists
  - function_call: Structured function calls (tool calling)
  - code_review: Code review feedback
  - search_results: Structured search results

CLI:
    lfm-train --dataset sahil2801/CodeAlpaca-20k \
        --structured-output \
        --benchmarks json_output toolcall \
        --hub-repo your-username/lfm-json-master
"""

from lfm_trainer.config import TrainingConfig
from lfm_trainer.trainer import run_training
from lfm_trainer.structured_output import (
    create_structured_output_dataset,
    validate_json,
    validate_against_schema,
    BUILTIN_SCHEMAS,
)
import json


# ═══════════════════════════════════════════════════════════════════════
#  Option A: Train with mixed structured + coding data
# ═══════════════════════════════════════════════════════════════════════
cfg_mixed = TrainingConfig(
    model_name="liquid/LFM2.5-1.2B-Base",
    dataset_paths=["sahil2801/CodeAlpaca-20k"],
    hub_repo_id="your-username/lfm-structured",

    # Enable structured output training data
    structured_output=True,

    # Training
    num_train_epochs=3,
    learning_rate=2e-4,
    use_lora=True,
    lora_r=32,
    bf16=True,

    # Benchmark with JSON output
    run_benchmark=True,
    benchmark_names=["json_output", "toolcall", "humaneval"],
    benchmark_before_after=True,
)
# run_training(cfg_mixed)


# ═══════════════════════════════════════════════════════════════════════
#  Option B: Generate and inspect the training data
# ═══════════════════════════════════════════════════════════════════════
def inspect_training_data():
    """Preview the structured output training examples."""
    dataset = create_structured_output_dataset(samples_per_schema=5)
    print(f"Generated {len(dataset)} training examples\n")

    # Show first example from each schema
    seen_schemas = set()
    for row in dataset:
        text = row["text"]
        # Extract schema name from the text
        for schema in BUILTIN_SCHEMAS:
            if schema["name"] not in seen_schemas and schema["name"] in text.lower():
                seen_schemas.add(schema["name"])
                print(f"{'='*60}")
                print(f"Schema: {schema['name']}")
                print(f"{'='*60}")
                print(text[:500])
                print("...\n")
                break

# inspect_training_data()


# ═══════════════════════════════════════════════════════════════════════
#  Option C: JSON validation utilities (use at inference time)
# ═══════════════════════════════════════════════════════════════════════
def test_validation():
    """Test JSON validation and schema conformance."""

    # Valid JSON matching person schema
    valid_response = '{"name": "Alice", "age": 28, "email": "alice@example.com", "skills": ["Python"]}'
    is_valid, obj, err = validate_json(valid_response)
    print(f"Valid JSON: {is_valid}")  # True

    # Validate against person schema
    person_schema = BUILTIN_SCHEMAS[0]["schema"]
    schema_valid, errors = validate_against_schema(valid_response, person_schema)
    print(f"Schema valid: {schema_valid}")  # True
    print(f"Errors: {errors}")  # []

    # Invalid JSON
    bad = '{"name": "Bob", age: 30}'  # Missing quotes on key
    is_valid, _, err = validate_json(bad)
    print(f"\nInvalid JSON: {is_valid}")  # False
    print(f"Error: {err}")

    # Missing required field
    missing = '{"age": 30}'  # Missing "name"
    _, errors = validate_against_schema(missing, person_schema)
    print(f"\nMissing required: {errors}")  # ["$.missing required field 'name'"]

# test_validation()


# ═══════════════════════════════════════════════════════════════════════
#  Option D: Custom schemas for your domain
# ═══════════════════════════════════════════════════════════════════════
custom_schemas = [
    {
        "name": "deployment",
        "description": "Deployment configuration",
        "schema": {
            "type": "object",
            "properties": {
                "service": {"type": "string"},
                "version": {"type": "string"},
                "replicas": {"type": "integer", "minimum": 1},
                "environment": {"type": "string", "enum": ["dev", "staging", "production"]},
                "config": {"type": "object"},
            },
            "required": ["service", "version", "environment"],
        },
    },
]

# Generate training data from custom schemas
# custom_dataset = create_structured_output_dataset(schemas=custom_schemas, samples_per_schema=50)
