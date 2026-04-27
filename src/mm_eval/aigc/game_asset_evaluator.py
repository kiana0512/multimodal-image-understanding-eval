"""Game asset evaluation helpers."""
from __future__ import annotations
import pandas as pd

def summarize_by_asset_type(df: pd.DataFrame, asset_type_col: str = "asset_type", score_col: str = "overall_score") -> pd.DataFrame:
    """Summarize quality scores by asset type."""
    if asset_type_col not in df.columns: raise ValueError(f"Missing column: {asset_type_col}")
    return df.groupby(asset_type_col)[score_col].agg(["count","mean","min","max"]).reset_index()
