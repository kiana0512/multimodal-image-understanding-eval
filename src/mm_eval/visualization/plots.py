"""Plot helpers."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

def plot_score_histogram(csv_path: str | Path, score_col: str, output_png: str | Path) -> None:
    """Plot a score histogram from CSV."""
    import matplotlib.pyplot as plt
    df=pd.read_csv(csv_path); p=Path(output_png); p.parent.mkdir(parents=True,exist_ok=True)
    df[score_col].hist(bins=20); plt.title(score_col); plt.xlabel(score_col); plt.ylabel("count"); plt.tight_layout(); plt.savefig(p); plt.close()
