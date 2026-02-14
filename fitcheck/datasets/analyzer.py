"""Basic local dataset analysis for fitcheck.

Analyzes JSONL/JSON files to detect format and estimate token counts
using a character-length heuristic (chars / 4). This is a rough proxy
that avoids a transformers dependency.

Does NOT:
- Tokenize with the model's actual tokenizer (Week 3)
- Fetch from HF Hub datasets (Week 3)
- Apply chat templates (Week 3)
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from fitcheck.models.profiles import DatasetProfile, SeqLenStats

# Character-to-token ratio heuristic.
# English text averages ~4 chars per token across most tokenizers.
# This is deliberately conservative (overestimates tokens → overestimates VRAM).
_CHARS_PER_TOKEN = 4

# Max rows to sample for statistics. We don't need all rows for p95/p99.
_MAX_SAMPLE_ROWS = 10_000


def analyze_local(path: str | Path, *, sample_limit: int = _MAX_SAMPLE_ROWS) -> DatasetProfile:
    """Analyze a local JSONL or JSON file.

    Args:
        path: Path to a .jsonl or .json file.
        sample_limit: Maximum number of rows to read for statistics.

    Returns:
        DatasetProfile with format detection and sequence length estimates.

    Raises:
        FileNotFoundError: If the path doesn't exist.
        ValueError: If the file format is not recognized.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    rows = _load_rows(filepath, sample_limit)
    if not rows:
        raise ValueError(f"No valid rows found in {filepath}")

    detected_format = _detect_format(rows[0])
    char_counts = [_count_text_chars(row, detected_format) for row in rows]
    token_estimates = [max(1, chars // _CHARS_PER_TOKEN) for chars in char_counts]
    token_estimates.sort()

    seq_len_stats = _compute_stats(token_estimates)

    return DatasetProfile(
        source=str(filepath),
        num_rows=len(rows),
        detected_format=detected_format,
        seq_len_stats=seq_len_stats,
    )


def _load_rows(filepath: Path, limit: int) -> list[dict]:
    """Load rows from a JSONL or JSON file."""
    suffix = filepath.suffix.lower()

    if suffix == ".jsonl":
        return _load_jsonl(filepath, limit)
    elif suffix == ".json":
        return _load_json(filepath, limit)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Expected .jsonl or .json")


def _load_jsonl(filepath: Path, limit: int) -> list[dict]:
    rows = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(rows) >= limit:
                break
    return rows


def _load_json(filepath: Path, limit: int) -> list[dict]:
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data[:limit]
    elif isinstance(data, dict):
        # Some datasets wrap rows in a key like "data" or "rows"
        for key in ("data", "rows", "train", "examples"):
            if key in data and isinstance(data[key], list):
                return data[key][:limit]
        return [data]
    else:
        raise ValueError(f"Expected JSON array or object, got {type(data).__name__}")


def _detect_format(row: dict) -> str:
    """Detect dataset format from column names."""
    keys = set(row.keys())

    # Alpaca format: instruction + output (input optional)
    if "instruction" in keys and "output" in keys:
        return "alpaca"

    # ShareGPT format: conversations list
    if "conversations" in keys and isinstance(row.get("conversations"), list):
        return "sharegpt"

    # Raw text: just a text field
    if "text" in keys:
        return "raw_text"

    return "unknown"


def _count_text_chars(row: dict, fmt: str) -> int:
    """Count total text characters in a row based on format."""
    if fmt == "alpaca":
        return sum(len(str(row.get(k, ""))) for k in ("instruction", "input", "output"))

    if fmt == "sharegpt":
        convos = row.get("conversations", [])
        return sum(len(str(turn.get("value", ""))) for turn in convos if isinstance(turn, dict))

    if fmt == "raw_text":
        return len(str(row.get("text", "")))

    # Unknown: concatenate all string values
    total = 0
    for v in row.values():
        if isinstance(v, str):
            total += len(v)
    return total


def _compute_stats(token_counts: list[int]) -> SeqLenStats:
    """Compute sequence length statistics from sorted token counts."""
    n = len(token_counts)
    return SeqLenStats(
        min=token_counts[0],
        mean=statistics.mean(token_counts),
        p50=token_counts[min(int(n * 0.50), n - 1)],
        p95=token_counts[min(int(n * 0.95), n - 1)],
        p99=token_counts[min(int(n * 0.99), n - 1)],
        max=token_counts[-1],
    )
