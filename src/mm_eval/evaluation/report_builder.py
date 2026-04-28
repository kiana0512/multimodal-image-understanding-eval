"""Markdown report builder."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import json
import pandas as pd

def _markdown_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as markdown without optional tabulate dependency."""
    if df.empty:
        return "Empty CSV."
    text_df = df.astype(str)
    columns = list(text_df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(row[col].replace("\n", " ") for col in columns) + " |")
    return "\n".join(lines)

def _table_from_csv(path: str | Path | None, max_rows: int = 10) -> str:
    if not path: return "Not provided."
    p=Path(path)
    if not p.exists(): return f"Not found: `{p}`"
    df=pd.read_csv(p)
    if df.empty: return "Empty CSV."
    return _markdown_table(df.head(max_rows))

def _json_metrics(path: str | Path | None) -> str:
    if not path:
        return "Not provided."
    p = Path(path)
    if not p.exists():
        return f"Not found: `{p}`"
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = [{"metric": key, "value": value} for key, value in data.items()]
    return _markdown_table(pd.DataFrame(rows))


def _figure_links(figures: list[str | Path] | None) -> str:
    """Render report figure links."""
    if not figures:
        return "Not provided."
    lines = []
    for figure in figures:
        path = Path(figure)
        if path.exists():
            lines.append(f"- `{path}`")
        else:
            lines.append(f"- Not found: `{path}`")
    return "\n".join(lines)


def build_markdown_report(title: str = "Experiment Report", retrieval_csv: str | Path | None = None, quality_csv: str | Path | None = None, segmentation_csv: str | Path | None = None, config_path: str | Path | None = None, retrieval_metrics_json: str | Path | None = None, figures: list[str | Path] | None = None) -> str:
    """Build a markdown report string from optional CSV outputs."""
    now=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""# {title}

Generated at: {now}

Status: placeholder / demo result unless explicitly replaced by real experiment logs.

## Config

`{config_path or 'not provided'}`

## Retrieval Results

{_table_from_csv(retrieval_csv)}

## Retrieval Metrics

{_json_metrics(retrieval_metrics_json)}

## Figures

{_figure_links(figures)}

## AIGC Quality Scores

{_table_from_csv(quality_csv)}

## Segmentation Metrics

{_table_from_csv(segmentation_csv)}

## Metric Explanation

- Retrieval metrics such as Recall@K measure whether the matched sample appears in the Top-K results.
- AIGC quality scores are heuristic proxy scores, not an industrial aesthetic model.
- Dice and IoU are used to evaluate segmentation and pseudo-label quality.

## Next Steps

- Replace toy data with real validation data.
- Add human review labels for badcase calibration.
- Track prompts, model versions and failure reasons in a repeatable experiment log.
"""

def save_markdown_report(markdown: str, output_dir: str | Path = "outputs/reports", prefix: str = "experiment_report") -> Path:
    """Save markdown report with timestamp."""
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); p=out/f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"; p.write_text(markdown,encoding="utf-8"); return p
