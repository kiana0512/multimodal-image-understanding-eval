"""Classification augmentation helpers."""
from __future__ import annotations
from mm_eval.data.transforms import build_image_transform

def build_classification_transforms(image_size: int = 224):
    """Return train and val transforms."""
    return build_image_transform(image_size,train=True), build_image_transform(image_size,train=False)
