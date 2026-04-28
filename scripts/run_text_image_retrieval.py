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
from mm_eval.retrieval.topk_export import export_text_image_topk, export_topk
from mm_eval.evaluation.retrieval_metrics import summarize_retrieval, summarize_retrieval_by_positives


def _feature_path(cfg: dict, explicit_key: str, legacy_name_key: str, default_name: str) -> Path:
    """Resolve feature path from explicit or legacy config keys."""
    if cfg.get(explicit_key):
        return Path(cfg[explicit_key])
    return Path(cfg.get("feature_dir", "outputs/features")) / cfg.get(legacy_name_key, default_name)


def main() -> None:
    """Run text-to-image retrieval from cached features."""
    parser = argparse.ArgumentParser(description="Run text-image retrieval and export Top-K CSV.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.get("output_dir", "outputs/retrieval"))
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg["manifest_path"])
    image_col = cfg.get("image_column", "image_path")
    text_col = cfg.get("text_column", "text")
    image_id_col = cfg.get("image_id_column")
    caption_id_col = cfg.get("caption_id_column")
    image_features = np.load(_feature_path(cfg, "image_feature_path", "image_feature_name", "image_features.npy"))
    text_features = np.load(_feature_path(cfg, "text_feature_path", "text_feature_name", "text_features.npy"))
    scores = compute_text_to_image_scores(text_features, image_features)
    topk_path = Path(cfg.get("topk_csv", out / cfg.get("topk_output_name", "text_to_image_topk.csv")))
    metrics_path = Path(cfg.get("metrics_json", out / cfg.get("metrics_output_name", "retrieval_metrics.json")))

    if image_id_col and image_id_col in df.columns:
        gallery_df = df.drop_duplicates(subset=[image_id_col]).reset_index(drop=True)
        export_text_image_topk(
            scores,
            df,
            gallery_df,
            topk_path,
            k=int(cfg.get("top_k", 5)),
            text_col=text_col,
            image_col=image_col,
            image_id_col=image_id_col,
            caption_id_col=caption_id_col or "caption_id",
        )
        query_ids = df[image_id_col].astype(str).to_numpy()
        gallery_ids = gallery_df[image_id_col].astype(str).to_numpy()
        positives = query_ids[:, None] == gallery_ids[None, :]
        metrics = summarize_retrieval_by_positives(scores, positives, ks=(1, min(5, scores.shape[1]), min(10, scores.shape[1])))
    else:
        ids = df[image_col].astype(str).tolist()
        export_topk(scores, ids, ids, topk_path, k=int(cfg.get("top_k", 5)))
        metrics = summarize_retrieval(scores, list(range(len(df))), ks=(1, min(5, len(df))))
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved Top-K results to {topk_path}")
    print(f"Saved retrieval metrics to {metrics_path}")
    print(metrics)


if __name__ == "__main__":
    main()
