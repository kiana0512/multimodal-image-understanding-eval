"""Pseudo-label diagnostics for quality-aware filtering and weighting."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .segmentation_metrics import evaluate_mask_pair

def _markdown_table(df: pd.DataFrame) -> str:
    """Render markdown without requiring pandas' optional tabulate dependency."""
    if df.empty:
        return "Empty report."
    text_df = df.astype(str)
    columns = list(text_df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(row[col].replace("\n", " ") for col in columns) + " |")
    return "\n".join(lines)

def evaluate_pseudo_labels(manifest: str | Path, gt_col: str = "gt_mask_path", pseudo_col: str = "pred_mask_path", threshold: float = 0.5) -> pd.DataFrame:
    """Compare teacher pseudo masks with GT masks."""
    df=pd.read_csv(manifest); rows=[]
    for _,row in df.iterrows():
        metrics=evaluate_mask_pair(row[gt_col],row[pseudo_col],threshold); rows.append({**row.to_dict(),**metrics,"quality_score":metrics["dice"]})
    return pd.DataFrame(rows)

def assign_quality_bins(df: pd.DataFrame, score_col: str = "quality_score") -> pd.DataFrame:
    """Assign top/middle/bottom bins by quantiles."""
    out=df.copy(); q1,q2=out[score_col].quantile([1/3,2/3])
    out["quality_bin"]=out[score_col].map(lambda v: "top" if v>=q2 else ("middle" if v>=q1 else "bottom"))
    return out

def hard_filter(df: pd.DataFrame, min_score: float = 0.6, score_col: str = "quality_score") -> pd.DataFrame:
    """Keep pseudo-labels above threshold."""
    return df[df[score_col] >= min_score].reset_index(drop=True)

def add_soft_weights(df: pd.DataFrame, score_col: str = "quality_score", min_weight: float = 0.2) -> pd.DataFrame:
    """Create sample weights from quality scores."""
    out=df.copy(); out["sample_weight"]=out[score_col].clip(lower=min_weight,upper=1.0); return out

def export_pseudo_label_report(df: pd.DataFrame, output_md: str | Path) -> None:
    """Export markdown diagnostics report."""
    p=Path(output_md); p.parent.mkdir(parents=True,exist_ok=True)
    summary = df.describe(include="all").reset_index().rename(columns={"index": "stat"})
    p.write_text("# Pseudo-label Diagnostics\n\nStatus: demo result / not yet run on real paper data.\n\n"+_markdown_table(summary),encoding="utf-8")
