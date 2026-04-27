from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from mm_eval.utils.config import load_config
from mm_eval.evaluation.segmentation_metrics import evaluate_segmentation_manifest


def main() -> None:
    """Run segmentation mask quality evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate Dice/IoU/Precision/Recall for mask pairs.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.get("output_dir", "outputs/segmentation_eval"))
    out.mkdir(parents=True, exist_ok=True)
    output_name = cfg.get("metrics_output_name", "mask_metrics.csv")
    output_csv = out / output_name
    evaluate_segmentation_manifest(
        cfg["manifest_path"],
        output_csv,
        gt_col=cfg.get("gt_mask_column", "gt_mask_path"),
        pred_col=cfg.get("pred_mask_column", "pred_mask_path"),
        threshold=float(cfg.get("threshold", 0.5)),
    )
    print(f"Saved segmentation metrics to {output_csv}")


if __name__ == "__main__":
    main()
