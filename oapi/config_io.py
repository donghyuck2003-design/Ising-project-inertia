from __future__ import annotations
from pathlib import Path
from typing import Mapping, Any
import json
from .config import SolverConfig


def apply_overrides(cfg: SolverConfig, overrides: Mapping[str, Any]) -> SolverConfig:
    """Apply nested or dotted-key JSON overrides to SolverConfig in place."""
    def set_dotted(key: str, value: Any):
        parts = key.split('.')
        obj: Any = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        if not hasattr(obj, parts[-1]):
            raise KeyError(f"Unknown config field: {key}")
        setattr(obj, parts[-1], value)

    for key, value in overrides.items():
        if key in ("controller", "anneal") and isinstance(value, Mapping):
            for k2, v2 in value.items():
                set_dotted(f"{key}.{k2}", v2)
        else:
            set_dotted(key, value)
    return cfg


def load_overrides(path: str | Path | None) -> dict:
    if path is None:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # auto_tune writes a wrapper with best_overrides.
    if isinstance(data, dict) and "best_overrides" in data:
        return dict(data["best_overrides"])
    return dict(data)
