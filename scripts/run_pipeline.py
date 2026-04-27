from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "flickr30k_retrieval": {
        "manifest": ROOT / "data/processed/flickr30k/flickr30k_manifest.csv",
        "steps": [
            [sys.executable, "scripts/extract_clip_features.py", "--config", "configs/flickr30k_clip_retrieval.yaml"],
            [sys.executable, "scripts/run_text_image_retrieval.py", "--config", "configs/flickr30k_clip_retrieval.yaml"],
            [sys.executable, "scripts/generate_report.py", "--retrieval-csv", "outputs/retrieval/flickr30k_text_to_image_topk.csv"],
        ],
        "prepare_hint": "python scripts/prepare_flickr30k_hf.py --max-samples 200",
    },
    "oxford_pet_segmentation": {
        "manifest": ROOT / "data/processed/oxford_pet/oxford_pet_segmentation_manifest.csv",
        "steps": [
            [sys.executable, "scripts/run_segmentation_quality_eval.py", "--config", "configs/oxford_pet_segmentation_quality.yaml"],
            [sys.executable, "scripts/generate_report.py", "--segmentation-csv", "outputs/segmentation_eval/oxford_pet_mask_metrics.csv"],
        ],
        "prepare_hint": "python scripts/prepare_oxford_pet.py --max-samples 200 --make-pseudo-masks",
    },
    "diffusiondb_quality": {
        "manifest": ROOT / "data/processed/diffusiondb/diffusiondb_manifest.csv",
        "steps": [
            [sys.executable, "scripts/run_aigc_quality_eval.py", "--config", "configs/diffusiondb_aigc_quality_eval.yaml"],
            [sys.executable, "scripts/generate_report.py", "--quality-csv", "outputs/quality_eval/diffusiondb_aigc_quality_scores.csv"],
        ],
        "prepare_hint": "python scripts/prepare_diffusiondb_subset.py --max-samples 100",
    },
    "game_asset_quality": {
        "manifest": ROOT / "data/processed/game_assets/game_asset_manifest.csv",
        "steps": [
            [sys.executable, "scripts/run_aigc_quality_eval.py", "--config", "configs/game_asset_quality_eval.yaml"],
            [sys.executable, "scripts/generate_report.py", "--quality-csv", "outputs/quality_eval/game_asset_quality_scores.csv"],
        ],
        "prepare_hint": "python scripts/prepare_local_game_assets.py --image-root D:/your_comfyui_outputs --output data/processed/game_assets/game_asset_manifest.csv",
    },
}


def run_step(cmd: list[str]) -> None:
    """Run one pipeline command and fail loudly on errors."""
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    """Run a named experiment pipeline without automatically downloading data."""
    parser = argparse.ArgumentParser(description="Unified entry for small real-data experiment pipelines.")
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    args = parser.parse_args()
    task = TASKS[args.task]
    manifest = task["manifest"]
    if not manifest.exists():
        raise FileNotFoundError(
            f"Required manifest not found: {manifest}\n"
            f"Please prepare data first, for example:\n  {task['prepare_hint']}"
        )
    print(f"Running task: {args.task}")
    print(f"Using manifest: {manifest}")
    for cmd in task["steps"]:
        run_step(cmd)
    print(f"Task completed: {args.task}")


if __name__ == "__main__":
    main()
