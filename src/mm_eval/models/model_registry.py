"""Small model registry."""
from __future__ import annotations
from .image_backbone import build_image_backbone

def build_model(task: str, **kwargs):
    """Build a model by task name."""
    if task == "classification": return build_image_backbone(**kwargs)
    raise ValueError(f"Unsupported task: {task}")
