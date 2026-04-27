"""Feature cache helpers."""
from __future__ import annotations
from pathlib import Path
import numpy as np

def save_feature_cache(features: np.ndarray, path: str | Path) -> None:
    """Save NumPy features."""
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); np.save(p,features)

def load_feature_cache(path: str | Path) -> np.ndarray:
    """Load NumPy features."""
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Feature cache not found: {p}")
    return np.load(p)
