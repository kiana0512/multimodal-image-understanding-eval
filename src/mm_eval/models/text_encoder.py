"""Placeholder text encoder interfaces."""
from __future__ import annotations
import numpy as np

class HashingTextEncoder:
    """Tiny deterministic text encoder for tests and toy fallback."""
    def __init__(self, dim: int = 128): self.dim=dim
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts with a hashing trick. This is not semantic CLIP."""
        feats=np.zeros((len(texts),self.dim),dtype=np.float32)
        for i,text in enumerate(texts):
            for token in text.lower().split(): feats[i,hash(token)%self.dim]+=1.0
        norm=np.linalg.norm(feats,axis=1,keepdims=True); return feats/np.maximum(norm,1e-12)
