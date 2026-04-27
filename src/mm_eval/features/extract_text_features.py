"""Text feature extraction entrypoints."""
from __future__ import annotations
import numpy as np
from mm_eval.models.clip_encoder import ClipEncoder

def extract_text_features(texts: list[str], model_name: str = "ViT-B-32", batch_size: int = 32, device: str | None = None) -> np.ndarray:
    """Extract CLIP/OpenCLIP text features."""
    encoder=ClipEncoder(model_name=model_name,device=device); return encoder.encode_texts(texts,batch_size=batch_size)
