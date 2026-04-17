"""Utilities for scoring multimodal instruction-tuning data quality."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DatasetSample",
    "MultiModalDataLoader",
    "MultiModalQualityScorer",
    "prepare_runtime_environment",
    "read_jsonl",
    "set_random_seed",
    "write_jsonl",
]


def __getattr__(name: str) -> Any:
    if name in {"DatasetSample", "MultiModalDataLoader"}:
        module = import_module(".data_loader", __name__)
        return getattr(module, name)
    if name == "MultiModalQualityScorer":
        module = import_module(".quality_scorer", __name__)
        return getattr(module, name)
    if name in {"prepare_runtime_environment", "read_jsonl", "set_random_seed", "write_jsonl"}:
        module = import_module(".runtime", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
