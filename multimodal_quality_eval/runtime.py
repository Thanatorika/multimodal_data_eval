"""Runtime utilities shared by the research scripts."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch


def ensure_dir(path: Path | str) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def prepare_runtime_environment(cache_root: Path | str) -> Path:
    """Redirect third-party caches to a writable local directory."""

    cache_root = ensure_dir(cache_root)
    hf_home = ensure_dir(cache_root / "huggingface")
    ensure_dir(hf_home / "hub")
    ensure_dir(hf_home / "datasets")
    ensure_dir(cache_root / "matplotlib")

    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return cache_root


def set_random_seed(seed: int) -> None:
    """Set seeds for deterministic-ish research runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {error}") from error
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number} of {path}.")
            records.append(payload)
    return records


def write_jsonl(path: Path | str, records: Iterable[dict[str, Any]]) -> Path:
    """Write a sequence of dictionaries to JSONL."""

    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def chunked(sequence: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
    """Yield fixed-size chunks from a sequence."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for index in range(0, len(sequence), batch_size):
        yield sequence[index : index + batch_size]


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a floating-point value to a closed interval."""

    return max(minimum, min(maximum, value))
