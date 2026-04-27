"""Device selection helpers."""
from __future__ import annotations

def get_device(device: str | None = None) -> str:
    """Return requested device, or cuda when available."""
    if device and device != "auto": return device
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
