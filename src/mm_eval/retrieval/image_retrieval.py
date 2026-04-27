"""Image-image retrieval helpers."""
from __future__ import annotations
import numpy as np
from .similarity import pairwise_cosine_similarity

def compute_image_retrieval_scores(query_features: np.ndarray, gallery_features: np.ndarray) -> np.ndarray:
    """Compute image-to-image retrieval scores."""
    return pairwise_cosine_similarity(query_features, gallery_features)
