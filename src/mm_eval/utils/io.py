"""Small IO utilities."""
from __future__ import annotations
from pathlib import Path
import json
from typing import Any

def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory."""
    p = Path(path); p.mkdir(parents=True, exist_ok=True); return p

def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file."""
    rows=[]
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def write_jsonl(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write rows to JSONL."""
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+"\n")
