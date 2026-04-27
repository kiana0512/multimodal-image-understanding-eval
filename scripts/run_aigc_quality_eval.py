from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from mm_eval.utils.config import load_config
from mm_eval.aigc.quality_scorer import AIGCQualityScorer
from mm_eval.aigc.badcase_miner import mine_badcases, write_badcase_markdown


def main() -> None:
    """Run heuristic AIGC quality evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate AIGC image quality with heuristic proxy scores.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg.get("output_dir", "outputs/quality_eval"))
    out.mkdir(parents=True, exist_ok=True)

    score_name = cfg.get("score_output_name", "aigc_quality_scores.csv")
    badcase_name = cfg.get("badcase_output_name", "badcases.csv")
    report_name = cfg.get("badcase_report_name", "badcase_report.md")

    scorer = AIGCQualityScorer(
        image_column=cfg.get("image_column", "image_path"),
        prompt_column=cfg.get("prompt_column", "prompt"),
    )
    score_path = out / score_name
    scores = scorer.score_manifest(cfg["manifest_path"], score_path)
    bad = mine_badcases(score_path, out / badcase_name)
    write_badcase_markdown(bad, out / report_name)
    print(f"Saved {len(scores)} quality rows to {score_path}. Scores are heuristic demo/proxy values.")


if __name__ == "__main__":
    main()
