"""Dataset validation helpers."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise ValueError when required columns are missing."""
    missing=[c for c in columns if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")

def require_existing_files(paths: list[str], root: str | Path = ".") -> None:
    """Raise FileNotFoundError for missing files."""
    base=Path(root); missing=[p for p in paths if not (base/p).exists()]
    if missing: raise FileNotFoundError(f"Missing {len(missing)} files, examples: {missing[:5]}")
