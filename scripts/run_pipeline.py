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
                "--csv",
                "outputs/retrieval/caption_text_to_image_topk.csv",
                "--image-column",
                "image_path",
                "--caption-column",
                "query_text",
                "--output",
                "outputs/figures/caption_topk_contact_sheet.png",
            ],
            [
                sys.executable,
                "scripts/generate_report.py",
                "--retrieval-csv",
                "outputs/retrieval/caption_text_to_image_topk.csv",
                "--retrieval-metrics-json",
                "outputs/retrieval/caption_retrieval_metrics.json",
            ],
        ],
        "prepare_hint": "python scripts/prepare_caption_dataset.py --source hf --dataset lambda/naruto-blip-captions --max-samples 500 --force",
        "missing_help": """
Please prepare a real image-caption dataset first.

Recommended default HuggingFace Parquet dataset:
python scripts/prepare_caption_dataset.py --source hf --dataset lambda/naruto-blip-captions --max-samples 500 --force

Optional ModelScope metadata/url sample, not recommended as default benchmark:
python scripts/prepare_caption_dataset.py --source modelscope --dataset coco_captions_small_slice --max-samples 500

Then run:
python scripts/run_pipeline.py --task caption_retrieval
""",
    },
    "oxford_pet_segmentation": {
        "manifest": ROOT / "data/processed/oxford_pet/oxford_pet_segmentation_manifest.csv",
        "steps": [
            [sys.executable, "scripts/run_segmentation_quality_eval.py", "--config", "configs/oxford_pet_segmentation_quality.yaml"],
            [sys.executable, "scripts/generate_report.py", "--segmentation-csv", "outputs/segmentation_eval/oxford_pet_mask_metrics.csv"],
        ],
        "prepare_hint": "python scripts/prepare_oxford_pet.py --max-samples 200 --make-pseudo-masks",
        "missing_help": """
This means Oxford Pet has not been prepared yet.

Option A: real dataset, requires network or local raw data
python scripts/prepare_oxford_pet.py --source auto --max-samples 200 --make-pseudo-masks
python scripts/run_pipeline.py --task oxford_pet_segmentation

If download fails with SSL/URLError:
- use --source local after manually extracting Oxford Pet into data/raw/oxford_pet
- or see README.md
""",
    },
    "diffusiondb_quality": {
        "manifest": ROOT / "data/processed/diffusiondb/diffusiondb_manifest.csv",
        "steps": [
            [sys.executable, "scripts/run_aigc_quality_eval.py", "--config", "configs/diffusiondb_aigc_quality_eval.yaml"],
            [sys.executable, "scripts/generate_report.py", "--quality-csv", "outputs/quality_eval/diffusiondb_aigc_quality_scores.csv"],
        ],
        "prepare_hint": "python scripts/prepare_diffusiondb_subset.py --mode local-images --local-root data/raw/diffusiondb_sample --max-samples 100",
        "missing_help": """
This means a local DiffusionDB image quality manifest has not been prepared yet.

Default DiffusionDB mode is metadata-only prompt analysis, not image quality evaluation:
python scripts/prepare_diffusiondb_subset.py --mode metadata-only --max-samples 1000
python scripts/analyze_prompts.py --manifest data/processed/diffusiondb/diffusiondb_metadata_manifest.csv --text-column prompt

For image quality evaluation, use local images:
python scripts/prepare_diffusiondb_subset.py --mode local-images --local-root data/raw/diffusiondb_sample --max-samples 100
python scripts/run_pipeline.py --task diffusiondb_quality

Recommended AIGC quality default is local ComfyUI/SD output:
python scripts/prepare_local_game_assets.py --image-root D:/RT/game-aigc-asset-workflow/outputs --output data/processed/game_assets/game_asset_manifest.csv
python scripts/run_pipeline.py --task game_asset_quality

DiffusionDB 2M/Large is very large and is not downloaded by default.
""",
    },
    "game_asset_quality": {
        "manifest": ROOT / "data/processed/game_assets/game_asset_manifest.csv",
        "steps": [
            [sys.executable, "scripts/run_aigc_quality_eval.py", "--config", "configs/game_asset_quality_eval.yaml"],
            [sys.executable, "scripts/generate_report.py", "--quality-csv", "outputs/quality_eval/game_asset_quality_scores.csv"],
        ],
        "prepare_hint": "python scripts/prepare_local_game_assets.py --image-root D:/your_comfyui_outputs --output data/processed/game_assets/game_asset_manifest.csv",
        "missing_help": """
This means the local game asset manifest has not been prepared yet.

Option A: scan your local ComfyUI/Stable Diffusion output directory
python scripts/prepare_local_game_assets.py --image-root D:/your_comfyui_outputs --output data/processed/game_assets/game_asset_manifest.csv
python scripts/run_pipeline.py --task game_asset_quality

Replace D:/your_comfyui_outputs with your own local image directory.
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
            "Unified entry for the main experiment pipelines: caption_retrieval, "
            "oxford_pet_segmentation, game_asset_quality, diffusiondb_quality."
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
