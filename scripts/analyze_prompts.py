from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

STYLE_WORDS = {
    "anime", "realistic", "cinematic", "fantasy", "pixel", "concept", "portrait",
    "illustration", "watercolor", "3d", "render", "icon", "logo", "ui",
}


def infer_asset_type(prompt: str) -> str:
    text = prompt.lower()
    if any(k in text for k in ["icon", "logo", "ui", "button"]):
        return "ui_icon"
    if any(k in text for k in ["character", "girl", "boy", "person", "warrior"]):
        return "character"
    if any(k in text for k in ["scene", "landscape", "city", "room"]):
        return "scene"
    return "general"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "Empty."
    text_df = df.astype(str)
    cols = list(text_df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in text_df.iterrows():
        lines.append("| " + " | ".join(row[c].replace("\n", " ") for c in cols) + " |")
    return "\n".join(lines)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]*", text.lower()) if len(t) >= 3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze prompt-only metadata manifests.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--text-column", default="prompt")
    parser.add_argument("--output-dir", default="outputs/prompt_analysis")
    parser.add_argument("--report-dir", default="outputs/reports")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if args.text_column not in df.columns:
        raise ValueError(f"Missing text column: {args.text_column}. Available columns: {list(df.columns)}")
    prompts = df[args.text_column].fillna("").astype(str)
    lengths = prompts.map(len)
    words = Counter(token for prompt in prompts for token in tokenize(prompt))
    style_counts = Counter(token for token in words if token in STYLE_WORDS)
    asset_counts = prompts.map(infer_asset_type).value_counts().reset_index()
    asset_counts.columns = ["asset_type", "count"]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats = pd.DataFrame([{
        "num_prompts": len(prompts),
        "avg_length": float(lengths.mean()) if len(lengths) else 0.0,
        "min_length": int(lengths.min()) if len(lengths) else 0,
        "max_length": int(lengths.max()) if len(lengths) else 0,
        "short_prompts_lt_20": int((lengths < 20).sum()),
        "long_prompts_gt_300": int((lengths > 300).sum()),
        "image_nsfw_count": int(pd.to_numeric(df.get("image_nsfw", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "image_nsfw" in df.columns else "",
        "prompt_nsfw_count": int(pd.to_numeric(df.get("prompt_nsfw", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if "prompt_nsfw" in df.columns else "",
    }])
    word_freq = pd.DataFrame(words.most_common(100), columns=["word", "count"])
    style_freq = pd.DataFrame(style_counts.most_common(), columns=["style_word", "count"])
    stats.to_csv(out / "prompt_stats.csv", index=False)
    word_freq.to_csv(out / "prompt_word_freq.csv", index=False)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"prompt_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report.write_text(
        "# Prompt Analysis Report\n\n"
        "Sensitive prompt text is not printed in this report; only aggregate statistics are shown.\n\n"
        "## Stats\n\n"
        f"{markdown_table(stats)}\n\n"
        "## Top Words\n\n"
        f"{markdown_table(word_freq.head(30))}\n\n"
        "## Style Words\n\n"
        f"{markdown_table(style_freq)}\n\n"
        "## Asset Type Guess\n\n"
        f"{markdown_table(asset_counts)}\n",
        encoding="utf-8",
    )
    print(f"Saved prompt stats: {out / 'prompt_stats.csv'}")
    print(f"Saved word frequency: {out / 'prompt_word_freq.csv'}")
    print(f"Saved report: {report}")


if __name__ == "__main__":
    main()
