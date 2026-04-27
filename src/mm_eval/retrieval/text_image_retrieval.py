"""Text-image retrieval helpers."""
from __future__ import annotations
import numpy as np
from .similarity import pairwise_cosine_similarity

def compute_text_to_image_scores(text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
    """Return text-query to image-gallery score matrix."""
    return pairwise_cosine_similarity(text_features, image_features)

def compute_image_to_text_scores(image_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    """Return image-query to text-gallery score matrix."""
    return pairwise_cosine_similarity(image_features, text_features)
