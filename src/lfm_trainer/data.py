"""
Structured dataset loading pipeline.

Supports multiple source formats (CSV, Parquet, JSONL, HuggingFace Hub IDs,
and direct ``pd.DataFrame`` objects) and merges them into a single unified
``datasets.Dataset`` ready for training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)


# ── Column mapping presets ─────────────────────────────────────────────────
# Map well-known dataset column layouts to a unified {"text": ...} format.
# Users can extend this dict or pass a custom formatter.
COLUMN_PRESETS: dict[str, dict[str, str]] = {
    # HF dataset: iamtarun/python_code_instructions_18k_alpaca
    "alpaca": {
        "instruction": "instruction",
        "input": "input",
        "output": "output",
    },
    # Generic prompt / response
    "prompt_response": {
        "prompt": "prompt",
        "response": "response",
    },
    # Conversational / chat / DataClaw style (e.g. peteromallet/dataclaw-peteromallet)
    "conversation": {
        "messages": "messages",
    },
    # Single-column already formatted
    "text": {
        "text": "text",
    },
}


def _format_alpaca(row: dict) -> dict:
    """Format an Alpaca-style row into a single ``text`` field."""
    instruction = row.get("instruction", "")
    inp = row.get("input", "")
    output = row.get("output", "")

    if inp:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{inp}\n\n"
            f"### Response:\n{output}"
        )
    else:
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
    return {"text": text}


def _format_prompt_response(row: dict) -> dict:
    """Format a prompt/response row into a single ``text`` field."""
    prompt = row.get("prompt", "")
    response = row.get("response", "")
    return {"text": f"### Prompt:\n{prompt}\n\n### Response:\n{response}"}


def _format_generic(row: dict, text_column: str = "text") -> dict:
    """Pass-through for datasets that already have a ``text`` column."""
    return {"text": row.get(text_column, "")}


def _format_messages(row: dict) -> dict:
    """Format a conversational / chat dataset into a single ``text`` field.

    Uses LFM 2.5's native special tokens for tool calling:
      - ``<|tool_call_start|>`` / ``<|tool_call_end|>`` for function calls
      - ``<|tool_result_start|>`` / ``<|tool_result_end|>`` for tool results

    Handles multiple conversation styles:
      - **DataClaw** (``peteromallet/dataclaw-peteromallet``): ``tool_uses`` list in assistant msgs
      - **OpenAI-style**: ``tool_calls`` list with ``function.name`` / ``function.arguments``
      - **ShareGPT / generic chat**: plain ``role`` / ``content`` messages
      - **Tool role**: messages with ``role="tool"`` carrying tool execution results
    """
    messages = row.get("messages", [])
    if not messages:
        return {"text": ""}

    # If the row has a top-level tools/functions list, prepend them as a system tool definition
    available_tools = row.get("tools", row.get("functions", []))

    parts: list[str] = []

    # ── Prepend available tool definitions if present ──────────────────
    if available_tools:
        import json

        tool_defs = json.dumps(available_tools, indent=2, default=str)
        parts.append(f"### System:\nYou have access to the following tools:\n```json\n{tool_defs}\n```")

    # ── Process each message ──────────────────────────────────────────
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "") or ""

        # Include reasoning traces — supports both 'think' (TxT360) and 'thinking' (DataClaw)
        reasoning = msg.get("think", "") or msg.get("thinking", "")

        if role == "tool":
            # Tool result message — wrap in LFM result tokens
            tool_name = msg.get("name", msg.get("tool_call_id", "tool"))
            block = (
                f"### Tool Result ({tool_name}):\n"
                f"<|tool_result_start|>\n{content}\n<|tool_result_end|>"
            )

        elif role == "assistant":
            # Build assistant block: <think> BEFORE content (think-then-act)
            parts_inner = []

            # Reasoning block (placed first — model thinks before acting)
            if reasoning:
                parts_inner.append(f"<think>\n{reasoning}\n</think>")

            # Main content
            if content:
                parts_inner.append(content)

            block = f"### Assistant:\n" + "\n\n".join(parts_inner) if parts_inner else "### Assistant:"

            # ── Tool calls: OpenAI-style (tool_calls list) ────────────
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    fn = tc.get("function", tc)
                    fn_name = fn.get("name", "unknown")
                    fn_args = fn.get("arguments", "")
                    # Use LFM 2.5 native Pythonic format
                    tc_parts.append(
                        f"<|tool_call_start|>\n{fn_name}({fn_args})\n<|tool_call_end|>"
                    )
                block += "\n\n" + "\n".join(tc_parts)

            # ── Tool uses: DataClaw-style (tool_uses list) ────────────
            tool_uses = msg.get("tool_uses", [])
            if tool_uses:
                tc_parts = []
                for tu in tool_uses:
                    tool_name = tu.get("tool", "unknown")
                    tool_input = tu.get("input", "")
                    tc_parts.append(
                        f"<|tool_call_start|>\n{tool_name}({tool_input!r})\n<|tool_call_end|>"
                    )
                block += "\n\n" + "\n".join(tc_parts)

        else:
            # user, system, or any other role
            block = f"### {role.capitalize()}:\n{content}"

        parts.append(block)

    return {"text": "\n\n".join(parts)}


# ── Public API ─────────────────────────────────────────────────────────────

def _load_single_source(path: str, text_column: str = "text") -> Dataset:
    """Load a single dataset source.

    Supports:
      - Local files: .csv, .parquet, .jsonl / .json
      - HuggingFace Hub dataset IDs (e.g. ``iamtarun/python_code_instructions_18k_alpaca``)
    """
    p = Path(path)

    if p.exists():
        suffix = p.suffix.lower()
        logger.info("Loading local file: %s (format=%s)", path, suffix)
        if suffix == ".csv":
            df = pd.read_csv(path)
        elif suffix == ".parquet":
            df = pd.read_parquet(path)
        elif suffix in {".jsonl", ".json"}:
            df = pd.read_json(path, lines=suffix == ".jsonl")
        else:
            raise ValueError(f"Unsupported local file format: {suffix}")
        ds = Dataset.from_pandas(df)
    else:
        # Treat as a HuggingFace Hub dataset ID
        logger.info("Loading HuggingFace dataset: %s", path)
        ds = load_dataset(path, split="train")

    return _apply_formatters(ds, text_column)


def _apply_formatters(ds: Dataset, text_column: str = "text") -> Dataset:
    """Auto-detect the column layout and convert to a unified ``{text}`` column."""
    columns = set(ds.column_names)

    if {"instruction", "output"}.issubset(columns):
        logger.info("Detected Alpaca format → applying formatter")
        ds = ds.map(_format_alpaca, remove_columns=ds.column_names)
    elif {"prompt", "response"}.issubset(columns):
        logger.info("Detected prompt/response format → applying formatter")
        ds = ds.map(_format_prompt_response, remove_columns=ds.column_names)
    elif "messages" in columns:
        logger.info("Detected conversational/chat format (e.g. DataClaw) → applying formatter")
        ds = ds.map(_format_messages, remove_columns=ds.column_names)
    elif text_column in columns:
        logger.info("Using existing '%s' column", text_column)
        ds = ds.map(
            lambda row: _format_generic(row, text_column),
            remove_columns=ds.column_names,
        )
    else:
        raise ValueError(
            f"Cannot auto-detect format.  Columns found: {columns}.  "
            f"Expected one of: alpaca (instruction/output), prompt/response, "
            f"conversational (messages), or a '{text_column}' column."
        )
    return ds


# ── Tool-calling filter ────────────────────────────────────────────────────

# Patterns that indicate a sample contains tool calls
_TOOL_CALL_MARKERS = [
    "<|tool_call_start|>",
    "<|tool_call_end|>",
    "tool_call",
    "function_call",
    '"tool_calls"',
    "tool_use",
    "<tool_call>",
    "<|python_tag|>",
]


def filter_tool_calling_only(ds: Dataset) -> Dataset:
    """Keep only rows whose ``text`` field contains tool-calling patterns.

    This enables training exclusively on tool-calling examples, which is
    useful for datasets like ``jdaddyalbs/playwright-mcp-toolcalling`` or
    any mixed dataset where you want to focus the model on learning
    when and how to invoke tools.
    """
    original_len = len(ds)

    def _has_tool_calls(row: dict) -> bool:
        text = row.get("text", "")
        return any(marker in text for marker in _TOOL_CALL_MARKERS)

    ds = ds.filter(_has_tool_calls)
    logger.info(
        "Tool-calling filter: kept %d / %d rows (%.1f%%)",
        len(ds), original_len, 100 * len(ds) / max(original_len, 1),
    )

# ── Data quality filters ──────────────────────────────────────────────────

def clean_dataset(
    ds: Dataset,
    min_length: int = 10,
    max_length: int = 100_000,
    deduplicate: bool = True,
) -> Dataset:
    """Apply data quality filters to a dataset.

    - Remove empty / whitespace-only rows
    - Remove rows shorter than ``min_length`` or longer than ``max_length``
    - Deduplicate by text hash
    """
    original_len = len(ds)

    # 1. Remove empty / whitespace-only
    ds = ds.filter(lambda row: bool(row.get("text", "").strip()))
    after_empty = len(ds)

    # 2. Length filter
    ds = ds.filter(lambda row: min_length <= len(row["text"]) <= max_length)
    after_length = len(ds)

    # 3. Deduplicate
    if deduplicate:
        seen: set[int] = set()
        keep_indices = []
        for i, row in enumerate(ds):
            h = hash(row["text"])
            if h not in seen:
                seen.add(h)
                keep_indices.append(i)
        ds = ds.select(keep_indices)

    logger.info(
        "Data quality: %d → %d rows (empty: -%d, length: -%d, dedup: -%d)",
        original_len,
        len(ds),
        original_len - after_empty,
        after_empty - after_length,
        after_length - len(ds) if deduplicate else 0,
    )
    return ds


def load_datasets(
    sources: list[Union[str, pd.DataFrame]],
    text_column: str = "text",
    tool_calling_only: bool = False,
    quality_filter: bool = False,
    min_length: int = 10,
    max_length: int = 100_000,
    eval_split: float = 0.0,
) -> Dataset | tuple[Dataset, Dataset]:
    """Load and merge multiple dataset sources into one unified Dataset.

    Parameters
    ----------
    sources:
        A list where each item is one of:
        - A local file path (CSV / Parquet / JSONL)
        - A HuggingFace Hub dataset ID (e.g. ``"iamtarun/python_code_instructions_18k_alpaca"``)
        - A ``pd.DataFrame`` already loaded in memory
    text_column:
        Column name to treat as the text column for generic/single-column datasets.
    tool_calling_only:
        If True, filter the final dataset to keep only rows that contain
        tool-calling patterns (e.g. ``<|tool_call_start|>``, ``tool_calls``, etc.).
    quality_filter:
        If True, remove empty rows, length outliers, and duplicates.
    min_length:
        Minimum text length to keep (only when quality_filter=True).
    max_length:
        Maximum text length to keep (only when quality_filter=True).
    eval_split:
        Fraction of data to hold out for evaluation (0.0–1.0). If > 0,
        returns a tuple of ``(train_dataset, eval_dataset)``.

    Returns
    -------
    A single ``datasets.Dataset`` with a ``text`` column, or a tuple of
    ``(train, eval)`` if ``eval_split > 0``.
    """
    if not sources:
        raise ValueError("At least one dataset source must be provided.")

    parts: list[Dataset] = []
    for idx, source in enumerate(sources):
        if isinstance(source, pd.DataFrame):
            logger.info("Loading DataFrame source #%d (%d rows)", idx, len(source))
            ds = _apply_formatters(Dataset.from_pandas(source), text_column)
        else:
            ds = _load_single_source(source, text_column)
        logger.info("  └─ %s rows from source #%d", len(ds), idx)
        parts.append(ds)

    merged = concatenate_datasets(parts)
    logger.info("Total merged dataset size: %s rows", len(merged))

    if tool_calling_only:
        merged = filter_tool_calling_only(merged)

    if quality_filter:
        merged = clean_dataset(merged, min_length=min_length, max_length=max_length)

    if eval_split > 0:
        split = merged.train_test_split(test_size=eval_split, seed=42)
        logger.info(
            "Train/eval split: %d train, %d eval (%.0f%%)",
            len(split["train"]), len(split["test"]), eval_split * 100,
        )
        return split["train"], split["test"]

    return merged

