"""Consistency checks for prompt-image and asset variants."""
from __future__ import annotations
import numpy as np
import pandas as pd
from mm_eval.retrieval.similarity import pairwise_cosine_similarity

def prompt_image_consistency(clip_scores: list[float], threshold: float = 0.25) -> pd.DataFrame:
    """Return consistency labels from prompt-image scores."""
    return pd.DataFrame({"consistency_score":clip_scores,"is_low_consistency":[s<threshold for s in clip_scores]})

def asset_style_consistency(df: pd.DataFrame, features: np.ndarray, asset_id_col: str = "asset_id") -> pd.DataFrame:
    """Compute mean intra-asset similarity for each asset_id."""
    rows=[]
    for asset_id,group in df.groupby(asset_id_col):
        idx=group.index.to_numpy(); sims=pairwise_cosine_similarity(features[idx],features[idx])
        score=float(sims[np.triu_indices_from(sims,k=1)].mean()) if len(idx)>1 else 1.0
        rows.append({"asset_id":asset_id,"consistency_score":score,"num_images":len(idx)})
    return pd.DataFrame(rows)

def multiview_3d_consistency_placeholder() -> dict[str,str]:
    """Reserved interface for future multi-view 3D render consistency."""
    return {"status":"placeholder","note":"future extension for 3D asset multi-view rendered images"}
