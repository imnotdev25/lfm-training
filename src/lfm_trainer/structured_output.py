"""
Structured Output Training — teach models to generate valid JSON conforming to schemas.

This module provides:
1. Training data generation from JSON schemas
2. Schema-augmented formatting (inject schema into prompts)
3. JSON validation utilities
4. A structured output benchmark

The training approach injects the JSON schema into the system/instruction prompt
and expects the assistant response to be valid JSON matching that schema.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from datasets import Dataset

logger = logging.getLogger(__name__)

# ── Built-in training schemas ─────────────────────────────────────────────

BUILTIN_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "person",
        "description": "Extract person information",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string", "format": "email"},
                "skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name"],
        },
    },
    {
        "name": "api_response",
        "description": "Format an API response",
        "schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "data": {"type": "object"},
                "message": {"type": "string"},
                "code": {"type": "integer"},
            },
            "required": ["status", "code"],
        },
    },
    {
        "name": "task_list",
        "description": "Create a structured task list",
        "schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "description": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                            "done": {"type": "boolean"},
                        },
                        "required": ["id", "description", "priority"],
                    },
                },
            },
            "required": ["title", "tasks"],
        },
    },
    {
        "name": "function_call",
        "description": "Structured function/tool call",
        "schema": {
            "type": "object",
            "properties": {
                "function": {"type": "string"},
                "arguments": {"type": "object"},
                "reasoning": {"type": "string"},
            },
            "required": ["function", "arguments"],
        },
    },
    {
        "name": "code_review",
        "description": "Structured code review feedback",
        "schema": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "line": {"type": "integer"},
                            "severity": {"type": "string", "enum": ["info", "warning", "error"]},
                            "message": {"type": "string"},
                            "suggestion": {"type": "string"},
                        },
                        "required": ["line", "severity", "message"],
                    },
                },
                "summary": {"type": "string"},
                "approved": {"type": "boolean"},
            },
            "required": ["file", "issues", "approved"],
        },
    },
    {
        "name": "search_results",
        "description": "Structured search results",
        "schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "total_results": {"type": "integer"},
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "url": {"type": "string"},
                            "snippet": {"type": "string"},
                            "relevance": {"type": "number"},
                        },
                        "required": ["title", "url"],
                    },
                },
            },
            "required": ["query", "results"],
        },
    },
]


# ── JSON Validation Utilities ─────────────────────────────────────────────


def validate_json(text: str) -> tuple[bool, Optional[Any], Optional[str]]:
    """Validate if text contains valid JSON.

    Returns (is_valid, parsed_obj, error_msg).
    """
    # Try to extract JSON from the text (may be wrapped in markdown code blocks)
    json_str = _extract_json(text)
    if not json_str:
        return False, None, "No JSON found in text"

    try:
        obj = json.loads(json_str)
        return True, obj, None
    except json.JSONDecodeError as e:
        return False, None, str(e)


def validate_against_schema(
    text: str,
    schema: dict[str, Any],
) -> tuple[bool, list[str]]:
    """Validate JSON text against a JSON schema.

    Returns (is_valid, list_of_errors).
    Uses lightweight validation (no jsonschema dependency required).
    """
    is_valid_json, obj, err = validate_json(text)
    if not is_valid_json:
        return False, [f"Invalid JSON: {err}"]

    errors = _validate_object(obj, schema, path="$")
    return len(errors) == 0, errors


def _extract_json(text: str) -> Optional[str]:
    """Extract JSON from text, handling markdown code blocks."""
    # Try direct parse first
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    # Try extracting from ```json ... ``` or ``` ... ```
    patterns = [
        r"```json\s*\n(.*?)\n\s*```",
        r"```\s*\n(.*?)\n\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()

    # Try finding JSON object/array in text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start_idx = text.find(start_char)
        if start_idx == -1:
            continue
        # Find matching end bracket
        depth = 0
        for i in range(start_idx, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx : i + 1]

    return None


def _validate_object(obj: Any, schema: dict, path: str = "$") -> list[str]:
    """Lightweight JSON Schema validation (no external dependency).

    Validates: type, required, enum, properties, items, minimum, maximum.
    """
    errors = []
    expected_type = schema.get("type")

    # Type check
    type_map = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }

    if expected_type and expected_type in type_map:
        expected = type_map[expected_type]
        if not isinstance(obj, expected):
            errors.append(f"{path}: expected {expected_type}, got {type(obj).__name__}")
            return errors  # Can't validate further on wrong type

    # Object validation
    if expected_type == "object" and isinstance(obj, dict):
        # Required fields
        for field in schema.get("required", []):
            if field not in obj:
                errors.append(f"{path}: missing required field '{field}'")

        # Property validation
        properties = schema.get("properties", {})
        for key, value in obj.items():
            if key in properties:
                errors.extend(_validate_object(value, properties[key], f"{path}.{key}"))

    # Array validation
    if expected_type == "array" and isinstance(obj, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(obj):
                errors.extend(_validate_object(item, items_schema, f"{path}[{i}]"))

    # Enum validation
    if "enum" in schema and obj not in schema["enum"]:
        errors.append(f"{path}: value '{obj}' not in enum {schema['enum']}")

    # Numeric bounds
    if "minimum" in schema and isinstance(obj, (int, float)):
        if obj < schema["minimum"]:
            errors.append(f"{path}: {obj} < minimum {schema['minimum']}")
    if "maximum" in schema and isinstance(obj, (int, float)):
        if obj > schema["maximum"]:
            errors.append(f"{path}: {obj} > maximum {schema['maximum']}")

    return errors


# ── Training Data Generation ──────────────────────────────────────────────


def create_structured_output_dataset(
    schemas: Optional[list[dict]] = None,
    samples_per_schema: int = 20,
) -> Dataset:
    """Create training examples that teach the model to output valid JSON.

    Each example has:
    - Instruction: a task + the JSON schema to follow
    - Response: valid JSON conforming to the schema

    Parameters
    ----------
    schemas : list of schema dicts, optional
        Each dict must have 'name', 'description', 'schema' keys.
        Defaults to BUILTIN_SCHEMAS.
    samples_per_schema : int
        Number of training examples per schema.
    """
    if schemas is None:
        schemas = BUILTIN_SCHEMAS

    examples = []
    for schema_def in schemas:
        name = schema_def["name"]
        desc = schema_def["description"]
        schema = schema_def["schema"]
        schema_str = json.dumps(schema, indent=2)

        # Generate diverse training examples for each schema
        generated = _generate_examples_for_schema(name, desc, schema, samples_per_schema)
        for prompt, response_json in generated:
            text = (
                f"### System:\n"
                f"You are a helpful assistant that responds ONLY with valid JSON.\n"
                f"Your response must conform to the following JSON schema:\n"
                f"```json\n{schema_str}\n```\n"
                f"Respond with ONLY the JSON object, no additional text.\n\n"
                f"### User:\n{prompt}\n\n"
                f"### Assistant:\n{response_json}"
            )
            examples.append({"text": text})

    logger.info("Created %d structured output training examples from %d schemas", len(examples), len(schemas))
    return Dataset.from_list(examples)


def _generate_examples_for_schema(
    name: str,
    description: str,
    schema: dict,
    count: int,
) -> list[tuple[str, str]]:
    """Generate (prompt, json_response) pairs for a given schema.

    Uses template-based generation with diverse prompts.
    """
    generators = {
        "person": _gen_person_examples,
        "api_response": _gen_api_response_examples,
        "task_list": _gen_task_list_examples,
        "function_call": _gen_function_call_examples,
        "code_review": _gen_code_review_examples,
        "search_results": _gen_search_results_examples,
    }

    generator = generators.get(name, _gen_generic_examples)
    return generator(schema, count)


def _gen_person_examples(schema: dict, count: int) -> list[tuple[str, str]]:
    names = [
        ("Alice Chen", 28, "alice@example.com", ["Python", "ML", "Docker"]),
        ("Bob Smith", 35, "bob@corp.io", ["Java", "Spring", "AWS"]),
        ("Carol Williams", 42, "carol@tech.dev", ["Rust", "Systems", "Linux"]),
        ("David Park", 24, "david@startup.co", ["JavaScript", "React", "Node.js"]),
        ("Eva Martinez", 31, "eva@uni.edu", ["R", "Statistics", "Data Science"]),
        ("Frank Lee", 29, "frank@dev.io", ["Go", "Kubernetes", "gRPC"]),
        ("Grace Kim", 38, "grace@ailab.org", ["PyTorch", "NLP", "Transformers"]),
        ("Henry Wang", 26, "henry@web.com", ["TypeScript", "Vue.js", "GraphQL"]),
        ("Iris Brown", 33, "iris@sec.io", ["Security", "Penetration Testing", "Cryptography"]),
        ("Jack Thompson", 45, "jack@enterprise.com", ["C++", "Embedded", "RTOS"]),
    ]
    prompts = [
        "Extract information about {}.",
        "Parse the following person's details: {}",
        "Create a structured profile for {}.",
        "Get the contact info for {}.",
    ]
    results = []
    for i in range(min(count, len(names) * len(prompts))):
        name_data = names[i % len(names)]
        prompt_tpl = prompts[i % len(prompts)]
        n, age, email, skills = name_data
        prompt = prompt_tpl.format(f"{n}, age {age}, email {email}, skills: {', '.join(skills)}")
        response = json.dumps({"name": n, "age": age, "email": email, "skills": skills}, indent=2)
        results.append((prompt, response))
    return results


def _gen_api_response_examples(schema: dict, count: int) -> list[tuple[str, str]]:
    cases = [
        ("User login successful", {"status": "success", "data": {"user_id": 123, "token": "abc123"}, "message": "Login successful", "code": 200}),
        ("File not found error", {"status": "error", "data": None, "message": "File '/data/report.csv' not found", "code": 404}),
        ("Created new resource", {"status": "success", "data": {"id": 456, "created_at": "2024-01-15T10:30:00Z"}, "message": "Resource created", "code": 201}),
        ("Rate limited", {"status": "error", "data": None, "message": "Too many requests. Retry after 60 seconds.", "code": 429}),
        ("Internal error", {"status": "error", "data": None, "message": "Internal server error", "code": 500}),
        ("Successfully deleted", {"status": "success", "data": {"deleted": True}, "message": "Resource deleted", "code": 200}),
        ("Unauthorized access", {"status": "error", "data": None, "message": "Invalid API key", "code": 401}),
        ("Validation error", {"status": "error", "data": {"fields": ["email", "name"]}, "message": "Validation failed", "code": 422}),
        ("Partial content", {"status": "success", "data": {"items": [1, 2, 3], "total": 100, "page": 1}, "message": "Page 1 of 10", "code": 206}),
        ("Service unavailable", {"status": "error", "data": None, "message": "Service under maintenance", "code": 503}),
    ]
    results = []
    for i in range(min(count, len(cases))):
        prompt, resp = cases[i % len(cases)]
        results.append((f"Format an API response for: {prompt}", json.dumps(resp, indent=2)))
    return results


def _gen_task_list_examples(schema: dict, count: int) -> list[tuple[str, str]]:
    task_lists = [
        ("Sprint Planning", [
            {"id": 1, "description": "Set up CI/CD pipeline", "priority": "high", "done": False},
            {"id": 2, "description": "Write unit tests for auth module", "priority": "medium", "done": False},
            {"id": 3, "description": "Update API documentation", "priority": "low", "done": True},
        ]),
        ("Bug Fixes", [
            {"id": 1, "description": "Fix login timeout on mobile", "priority": "high", "done": False},
            {"id": 2, "description": "Resolve memory leak in cache", "priority": "high", "done": False},
        ]),
        ("Weekend Chores", [
            {"id": 1, "description": "Grocery shopping", "priority": "high", "done": False},
            {"id": 2, "description": "Clean garage", "priority": "low", "done": False},
            {"id": 3, "description": "Fix leaky faucet", "priority": "medium", "done": False},
        ]),
    ]
    results = []
    for i in range(min(count, len(task_lists) * 3)):
        title, tasks = task_lists[i % len(task_lists)]
        results.append(
            (f"Create a task list for '{title}'", json.dumps({"title": title, "tasks": tasks}, indent=2))
        )
    return results


def _gen_function_call_examples(schema: dict, count: int) -> list[tuple[str, str]]:
    calls = [
        ("Get the weather in Tokyo", {"function": "get_weather", "arguments": {"location": "Tokyo", "unit": "celsius"}, "reasoning": "User wants current weather in Tokyo"}),
        ("Search for Python tutorials", {"function": "search_web", "arguments": {"query": "Python tutorials", "limit": 10}, "reasoning": "User is looking for learning resources"}),
        ("Send an email to team", {"function": "send_email", "arguments": {"to": "team@company.com", "subject": "Meeting Tomorrow", "body": "Reminder: team meeting at 2pm"}, "reasoning": "User wants to send a notification email"}),
        ("Create a new database entry", {"function": "db_insert", "arguments": {"table": "users", "data": {"name": "Alice", "role": "admin"}}, "reasoning": "User wants to add a new user record"}),
        ("Calculate compound interest", {"function": "calculate", "arguments": {"principal": 10000, "rate": 0.05, "years": 10, "compound": "monthly"}, "reasoning": "Financial calculation requested"}),
    ]
    results = []
    for i in range(min(count, len(calls) * 2)):
        prompt, resp = calls[i % len(calls)]
        results.append((prompt, json.dumps(resp, indent=2)))
    return results


def _gen_code_review_examples(schema: dict, count: int) -> list[tuple[str, str]]:
    reviews = [
        ("Review this Python function:\ndef add(a, b): return a + b", {
            "file": "math_utils.py",
            "issues": [
                {"line": 1, "severity": "info", "message": "Missing type hints", "suggestion": "def add(a: int, b: int) -> int:"},
                {"line": 1, "severity": "warning", "message": "No docstring", "suggestion": "Add a docstring explaining the function"},
            ],
            "summary": "Simple function, needs type hints and documentation",
            "approved": True,
        }),
        ("Review: conn = sqlite3.connect(user_input)", {
            "file": "database.py",
            "issues": [
                {"line": 1, "severity": "error", "message": "SQL injection risk: user input used directly in connection string", "suggestion": "Use parameterized queries and validate input"},
            ],
            "summary": "Critical security issue found",
            "approved": False,
        }),
    ]
    results = []
    for i in range(min(count, len(reviews) * 3)):
        prompt, resp = reviews[i % len(reviews)]
        results.append((prompt, json.dumps(resp, indent=2)))
    return results


def _gen_search_results_examples(schema: dict, count: int) -> list[tuple[str, str]]:
    searches = [
        ("machine learning tutorials", {
            "query": "machine learning tutorials",
            "total_results": 3,
            "results": [
                {"title": "ML Course by Andrew Ng", "url": "https://coursera.org/ml", "snippet": "Stanford's ML course", "relevance": 0.98},
                {"title": "Fast.ai Practical DL", "url": "https://fast.ai", "snippet": "Practical deep learning", "relevance": 0.95},
                {"title": "scikit-learn Docs", "url": "https://scikit-learn.org", "snippet": "Python ML library", "relevance": 0.90},
            ],
        }),
        ("rust web framework", {
            "query": "rust web framework",
            "total_results": 2,
            "results": [
                {"title": "Actix Web", "url": "https://actix.rs", "snippet": "Powerful Rust web framework", "relevance": 0.96},
                {"title": "Axum", "url": "https://github.com/tokio-rs/axum", "snippet": "Ergonomic web framework", "relevance": 0.94},
            ],
        }),
    ]
    results = []
    for i in range(min(count, len(searches) * 3)):
        query, resp = searches[i % len(searches)]
        results.append((f"Search for: {query}", json.dumps(resp, indent=2)))
    return results


def _gen_generic_examples(schema: dict, count: int) -> list[tuple[str, str]]:
    """Fallback generator for unknown schema types."""
    obj = _generate_default_from_schema(schema)
    response = json.dumps(obj, indent=2)
    return [(f"Generate a response matching the provided schema", response)] * min(count, 5)


def _generate_default_from_schema(schema: dict) -> Any:
    """Generate a default object that conforms to a JSON schema."""
    t = schema.get("type", "object")
    if t == "object":
        obj = {}
        for key, prop_schema in schema.get("properties", {}).items():
            obj[key] = _generate_default_from_schema(prop_schema)
        return obj
    if t == "array":
        items = schema.get("items", {"type": "string"})
        return [_generate_default_from_schema(items)]
    if t == "string":
        if "enum" in schema:
            return schema["enum"][0]
        return "example"
    if t == "integer":
        return schema.get("minimum", 1)
    if t == "number":
        return schema.get("minimum", 1.0)
    if t == "boolean":
        return True
    if t == "null":
        return None
    return "unknown"


# ── Schema-augmented Data Formatting ──────────────────────────────────────

def augment_with_schema(dataset: Dataset, schema: dict, schema_name: str = "output") -> Dataset:
    """Add JSON schema instructions to an existing dataset's prompts.

    Wraps existing instruction/prompt with schema requirement and
    reformats responses as JSON where possible.
    """
    schema_str = json.dumps(schema, indent=2)

    def _augment(row):
        text = row.get("text", "")
        # Inject schema instruction after system/instruction prompt
        schema_block = (
            f"\n\n**Output Format**: Respond with valid JSON matching this schema:\n"
            f"```json\n{schema_str}\n```\n"
        )
        # Insert before ### Assistant: or ### Response:
        for marker in ["### Assistant:", "### Response:"]:
            if marker in text:
                text = text.replace(marker, schema_block + "\n" + marker)
                break
        return {"text": text}

    return dataset.map(_augment)
