"""Export Top-K retrieval results."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .similarity import topk_indices

def export_topk(scores: np.ndarray, query_ids: list[str], gallery_ids: list[str], output_csv: str | Path, k: int = 5) -> pd.DataFrame:
    """Export Top-K gallery items for each query."""
    rows=[]; idx=topk_indices(scores,k=k)
    for qi,q in enumerate(query_ids):
        for rank,gi in enumerate(idx[qi], start=1):
            rows.append({"query_id":q,"rank":rank,"gallery_id":gallery_ids[int(gi)],"score":float(scores[qi,gi])})
    df=pd.DataFrame(rows); p=Path(output_csv); p.parent.mkdir(parents=True,exist_ok=True); df.to_csv(p,index=False); return df

def export_text_image_topk(
    scores: np.ndarray,
    query_df: pd.DataFrame,
    gallery_df: pd.DataFrame,
    output_csv: str | Path,
    k: int = 5,
    text_col: str = "text",
    image_col: str = "image_path",
    image_id_col: str = "image_id",
    caption_id_col: str = "caption_id",
) -> pd.DataFrame:
    """Export caption-query to image-gallery Top-K with image_id positive labels."""
    rows = []
    idx = topk_indices(scores, k=k)
    for qi, query in query_df.reset_index(drop=True).iterrows():
        query_image_id = str(query.get(image_id_col, qi))
        for rank, gi in enumerate(idx[qi], start=1):
            gallery = gallery_df.iloc[int(gi)]
            matched_image_id = str(gallery.get(image_id_col, gi))
            rows.append({
                "query_id": str(query.get(caption_id_col, qi)),
                "query_text": str(query.get(text_col, "")),
                "rank": rank,
                "image_path": str(gallery.get(image_col, "")),
                "matched_text": str(gallery.get(text_col, "")),
                "score": float(scores[qi, gi]),
                "is_positive": int(query_image_id == matched_image_id),
                "image_id": query_image_id,
                "matched_image_id": matched_image_id,
                "caption_id": str(query.get(caption_id_col, qi)),
            })
    df = pd.DataFrame(rows)
    p = Path(output_csv)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return df
