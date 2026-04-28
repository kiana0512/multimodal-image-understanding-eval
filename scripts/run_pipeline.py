from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

TASKS = {
    "caption_retrieval": {
        "manifest": ROOT / "data/processed/caption/caption_manifest.csv",
        "steps": [
            [sys.executable, "scripts/extract_clip_features.py", "--config", "configs/caption_clip_retrieval.yaml"],
            [sys.executable, "scripts/run_text_image_retrieval.py", "--config", "configs/caption_clip_retrieval.yaml"],
            [
                sys.executable,
                "scripts/make_contact_sheet.py",
                "--retrieval-csv",
                "outputs/retrieval/caption_text_to_image_topk.csv",
                "--output",
                "outputs/figures/caption_topk_contact_sheet.png",
                "--num-queries",
                "8",
                "--top-k",
                "5",
            ],
            [
                sys.executable,
                "scripts/generate_report.py",
                "--retrieval-csv",
                "outputs/retrieval/caption_text_to_image_topk.csv",
                "--retrieval-metrics-json",
                "outputs/retrieval/caption_retrieval_metrics.json",
                "--figures",
                "outputs/figures/caption_topk_contact_sheet.png",
            ],
        ],
        "prepare_hint": "python scripts/prepare_coco2014_caption.py --split val --max-samples 5000 --force",
        "missing_help": """
Please prepare a real image-caption dataset first.

Recommended official COCO2014 Caption workflow:
python scripts/download_coco2014_official.py --split val --download --extract
python scripts/prepare_coco2014_caption.py --split val --max-samples 5000 --force
python scripts/run_pipeline.py --task caption_retrieval
""",
    },
    "oxford_pet_segmentation": {
        "manifest": ROOT / "data/processed/oxford_pet/oxford_pet_segmentation_manifest.csv",
        "steps": [
            [sys.executable, "scripts/run_segmentation_quality_eval.py", "--config", "configs/oxford_pet_segmentation_quality.yaml"],
            [sys.executable, "scripts/generate_report.py", "--segmentation-csv", "outputs/segmentation_eval/oxford_pet_mask_metrics.csv"],
        ],
        "prepare_hint": "python scripts/prepare_oxford_pet.py --source auto --max-samples 500 --make-pseudo-masks --force",
        "missing_help": """
Recommended Oxford-IIIT Pet segmentation workflow:
python scripts/prepare_oxford_pet.py --source auto --max-samples 500 --make-pseudo-masks --force
python scripts/run_pipeline.py --task oxford_pet_segmentation

If the dataset is already extracted locally:
python scripts/prepare_oxford_pet.py --source local --local-root data/raw/oxford_pet --max-samples 500 --make-pseudo-masks --force
python scripts/run_pipeline.py --task oxford_pet_segmentation
""",
    },
    "game_asset_quality": {
        "manifest": ROOT / "data/processed/game_assets/game_asset_manifest.csv",
        "steps": [
            [sys.executable, "scripts/run_aigc_quality_eval.py", "--config", "configs/game_asset_quality_eval.yaml"],
            [sys.executable, "scripts/generate_report.py", "--quality-csv", "outputs/quality_eval/game_asset_quality_scores.csv"],
        ],
        "prepare_hint": "python scripts/prepare_local_game_assets.py --image-root D:/your_aigc_outputs --output data/processed/game_assets/game_asset_manifest.csv",
        "missing_help": """
Recommended local AIGC asset quality workflow:
python scripts/prepare_local_game_assets.py --image-root D:/your_aigc_outputs --output data/processed/game_assets/game_asset_manifest.csv
python scripts/run_pipeline.py --task game_asset_quality

Replace D:/your_aigc_outputs with your ComfyUI / Stable Diffusion / generated asset folder.
""",
    },
}


def run_step(cmd: list[str]) -> None:
    """Run one pipeline command and fail loudly on errors."""
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def rel(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_missing_manifest_message(task_name: str, task: dict) -> str:
    """Build a task-specific missing-manifest message."""
    manifest = task["manifest"]
    return (
        "Required manifest not found:\n"
        f"{rel(manifest)}\n\n"
        "run_pipeline.py does not download or prepare real datasets. It only runs an "
        "experiment after the required manifest already exists.\n\n"
        f"{task.get('missing_help') or task['prepare_hint']}\n"
        "\nDiagnostic helper:\n"
        "python scripts/data_doctor.py\n"
        f"\nTask: {task_name}"
    )


def main() -> None:
    """Run a named experiment pipeline without automatically downloading data."""
    parser = argparse.ArgumentParser(
        description=(
            "Unified entry for COCO2014 retrieval, Oxford Pet segmentation, and local AIGC asset quality pipelines."
        )
    )
    parser.add_argument("--task", required=True, choices=sorted(TASKS))
    args = parser.parse_args()
    task = TASKS[args.task]
    manifest = task["manifest"]
    if not manifest.exists():
        print(build_missing_manifest_message(args.task, task), file=sys.stderr)
        raise SystemExit(2)
    print(f"Running task: {args.task}")
    print(f"Using manifest: {manifest}")
    for cmd in task["steps"]:
        run_step(cmd)
    print(f"Task completed: {args.task}")


if __name__ == "__main__":
    main()
