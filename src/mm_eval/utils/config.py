"""YAML configuration helpers."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {p}")
    return data
