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
