"""Badcase mining for retrieval and AIGC quality outputs."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

BADCASE_QUERY = "badcase_reason != 'ok' or overall_score < 0.5"

def _markdown_table(df: pd.DataFrame) -> str:
    """Render markdown without requiring pandas' optional tabulate dependency."""
    if df.empty:
        return "No badcases found."
    text_df = df.astype(str)
    columns = list(text_df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(row[col].replace("\n", " ") for col in columns) + " |")
    return "\n".join(lines)

def mine_badcases(score_csv: str | Path, output_csv: str | Path, top_n: int = 50) -> pd.DataFrame:
    """Export low-score or tagged badcases."""
    df=pd.read_csv(score_csv)
    bad=df.query(BADCASE_QUERY).sort_values("overall_score",ascending=True).head(top_n) if "overall_score" in df.columns else df.head(top_n)
    p=Path(output_csv); p.parent.mkdir(parents=True,exist_ok=True); bad.to_csv(p,index=False); return bad

def write_badcase_markdown(badcases: pd.DataFrame, output_md: str | Path) -> None:
    """Write a markdown badcase report."""
    p=Path(output_md); p.parent.mkdir(parents=True,exist_ok=True)
    text="# Badcase Report\n\nStatus: demo result / placeholder until real data is evaluated.\n\n"+_markdown_table(badcases)
    p.write_text(text,encoding="utf-8")
