from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
import numpy as np
import pandas as pd
from mm_eval.utils.config import load_config
from mm_eval.retrieval.text_image_retrieval import compute_text_to_image_scores
from mm_eval.retrieval.topk_export import export_topk
from mm_eval.evaluation.retrieval_metrics import summarize_retrieval


def main() -> None:
    """Run text-to-image retrieval from cached features."""
    parser = argparse.ArgumentParser(description="Run text-image retrieval and export Top-K CSV.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.get("output_dir", "outputs/retrieval"))
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg["manifest_path"])
    feat_dir = Path(cfg.get("feature_dir", "outputs/features"))
    image_features = np.load(feat_dir / cfg.get("image_feature_name", "image_features.npy"))
    text_features = np.load(feat_dir / cfg.get("text_feature_name", "text_features.npy"))
    scores = compute_text_to_image_scores(text_features, image_features)
    ids = df[cfg.get("image_column", "image_path")].astype(str).tolist()
    topk_path = out / cfg.get("topk_output_name", "text_to_image_topk.csv")
    export_topk(scores, ids, ids, topk_path, k=int(cfg.get("top_k", 5)))
    metrics = summarize_retrieval(scores, list(range(len(df))), ks=(1, min(5, len(df))))
    metrics_path = out / cfg.get("metrics_output_name", "retrieval_metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved Top-K results to {topk_path}")
    print(metrics)


if __name__ == "__main__":
    main()
