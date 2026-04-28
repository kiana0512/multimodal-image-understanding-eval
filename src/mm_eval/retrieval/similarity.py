"""Similarity and Top-K utilities."""
from __future__ import annotations
import numpy as np

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize vectors along the last dimension."""
    arr=np.asarray(x,dtype=np.float32); norm=np.linalg.norm(arr,axis=-1,keepdims=True)
    return arr/np.maximum(norm,eps)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    aa=l2_normalize(np.asarray(a).reshape(1,-1))[0]; bb=l2_normalize(np.asarray(b).reshape(1,-1))[0]
    return float(np.dot(aa,bb))

def pairwise_cosine_similarity(image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    """Compute a pairwise cosine similarity matrix."""
    return l2_normalize(image_features) @ l2_normalize(text_features).T

def topk_indices(scores: np.ndarray, k: int = 5) -> np.ndarray:
    """Return descending Top-K indices for a vector or matrix."""
    arr=np.asarray(scores)
    k = min(k, arr.shape[-1])
    if arr.ndim == 1: return np.argsort(-arr)[:k]
    return np.argsort(-arr,axis=1)[:,:k]

def image_text_matching_score(image_feature: np.ndarray, text_feature: np.ndarray) -> float:
    """Return one image-text cosine matching score."""
    return cosine_similarity(image_feature,text_feature)
