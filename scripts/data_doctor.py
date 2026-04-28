from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def status(flag: bool) -> str:
    """Return a readable OK/MISSING status."""
    return "OK" if flag else "MISSING"


def found(flag: bool) -> str:
    """Return a readable FOUND/MISSING status."""
    return "FOUND" if flag else "MISSING"


def rel(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def csv_counts(path: Path) -> tuple[int | None, int | None]:
    """Return unique image count and row count for a manifest."""
    if not path.exists():
        return None, None
    try:
        import pandas as pd

        df = pd.read_csv(path)
        if "image_id" in df.columns:
            image_count = int(df["image_id"].nunique())
        elif "image_path" in df.columns:
            image_count = int(df["image_path"].nunique())
        else:
            image_count = None
        return image_count, int(len(df))
    except Exception:
        return None, None


def find_oxford_pet_root(root: Path) -> Path | None:
    """Find an extracted Oxford-IIIT Pet directory without downloading anything."""
    if not root.exists():
        return None
    candidates = [root]
    candidates.extend(
        p for p in root.rglob("*") if p.is_dir() and p.name.lower() in {"oxford-iiit-pet", "oxford_pet"}
    )
    for candidate in candidates:
        annotations = candidate / "annotations"
        if (
            (candidate / "images").is_dir()
            and (annotations / "trimaps").is_dir()
            and (annotations / "trainval.txt").is_file()
            and (annotations / "test.txt").is_file()
        ):
            return candidate
    return None


def print_environment() -> None:
    """Print lightweight environment checks."""
    print("Environment:")
    print(f"- Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"- Project root: {ROOT}")
    try:
        import mm_eval  # noqa: F401

        print("- mm_eval import: OK")
    except Exception as exc:
        print(f"- mm_eval import: FAIL ({exc})")


def print_dataset_status() -> None:
    """Print manifest and raw-data status."""
    raw_root = ROOT / "data/raw"
    processed_root = ROOT / "data/processed"

    print("\nDataset status:")
    print(f"- data/raw: {found(raw_root.exists())} ({rel(raw_root)})")
    print(f"- data/processed: {found(processed_root.exists())} ({rel(processed_root)})")
    print("\nOfficial COCO2014:")
    coco_root = ROOT / "data/raw/coco2014"
    coco_checks = [
        ("val2014.zip", coco_root / "val2014.zip"),
        ("train2014.zip", coco_root / "train2014.zip"),
        ("annotations_trainval2014.zip", coco_root / "annotations_trainval2014.zip"),
        ("val2014 extracted", coco_root / "val2014"),
        ("train2014 extracted", coco_root / "train2014"),
        ("captions_val2014.json", coco_root / "annotations/captions_val2014.json"),
        ("captions_train2014.json", coco_root / "annotations/captions_train2014.json"),
        ("caption manifest", ROOT / "data/processed/caption/caption_manifest.csv"),
    ]
    for name, path in coco_checks:
        print(f"- {name}: {found(path.exists())} ({rel(path)})")
    caption_manifest = ROOT / "data/processed/caption/caption_manifest.csv"
    image_count, pair_count = csv_counts(caption_manifest)
    if image_count is not None:
        print(f"- caption image count: {image_count}")
    if pair_count is not None:
        print(f"- caption pair count: {pair_count}")
    if image_count is not None and image_count < 20:
        print("  WARNING: fewer than 20 images is too small for a useful retrieval demo.")

    print("\nOxford-IIIT Pet segmentation:")
    oxford_root = find_oxford_pet_root(ROOT / "data/raw/oxford_pet")
    oxford_checks = [
        ("raw data", oxford_root),
        ("segmentation manifest", ROOT / "data/processed/oxford_pet/oxford_pet_segmentation_manifest.csv"),
        ("mask metrics CSV", ROOT / "outputs/segmentation_eval/oxford_pet_mask_metrics.csv"),
        ("mask metrics summary", ROOT / "outputs/segmentation_eval/oxford_pet_mask_metrics_summary.csv"),
    ]
    for name, path in oxford_checks:
        exists = path is not None and Path(path).exists()
        display = rel(Path(path)) if path is not None else "data/raw/oxford_pet"
        print(f"- {name}: {found(exists)} ({display})")

    print("\nLocal AIGC asset quality:")
    aigc_checks = [
        ("game asset manifest", ROOT / "data/processed/game_assets/game_asset_manifest.csv"),
        ("quality scores", ROOT / "outputs/quality_eval/game_asset_quality_scores.csv"),
        ("badcases CSV", ROOT / "outputs/quality_eval/game_asset_badcases.csv"),
        ("badcase report", ROOT / "outputs/quality_eval/game_asset_badcase_report.md"),
    ]
    for name, path in aigc_checks:
        print(f"- {name}: {found(path.exists())} ({rel(path)})")

    print("\nCaption retrieval outputs:")
    output_checks = [
        ("caption image features", ROOT / "outputs/features/caption_image_features.npy"),
        ("caption text features", ROOT / "outputs/features/caption_text_features.npy"),
        ("caption metadata", ROOT / "outputs/features/caption_metadata.csv"),
        ("caption Top-K CSV", ROOT / "outputs/retrieval/caption_text_to_image_topk.csv"),
        ("caption retrieval metrics JSON", ROOT / "outputs/retrieval/caption_retrieval_metrics.json"),
        ("caption contact sheet", ROOT / "outputs/figures/caption_topk_contact_sheet.png"),
    ]
    for name, path in output_checks:
        print(f"- {name}: {status(path.exists())} ({rel(path)})")


def print_recommendations() -> None:
    """Print Windows-friendly next commands."""
    print(
        "\nRecommended next commands:\n"
        "\n"
        "1. Quick COCO2014 val run:\n"
        "   python scripts/download_coco2014_official.py --split val --download --extract\n"
        "   python scripts/prepare_coco2014_caption.py --split val --max-samples 5000 --force\n"
        "   python scripts/run_pipeline.py --task caption_retrieval\n"
        "\n"
        "2. Full COCO2014 train+val run:\n"
        "   python scripts/download_coco2014_official.py --split trainval --download --extract\n"
        "   python scripts/prepare_coco2014_caption.py --split trainval --max-samples 20000 --force\n"
        "   python scripts/run_pipeline.py --task caption_retrieval\n"
        "\n"
        "3. Oxford-IIIT Pet segmentation:\n"
        "   python scripts/prepare_oxford_pet.py --source auto --max-samples 500 --make-pseudo-masks --force\n"
        "   python scripts/run_pipeline.py --task oxford_pet_segmentation\n"
        "\n"
        "4. Local AIGC asset quality:\n"
        "   python scripts/prepare_local_game_assets.py --image-root D:/your_aigc_outputs --output data/processed/game_assets/game_asset_manifest.csv\n"
        "   python scripts/run_pipeline.py --task game_asset_quality\n"
    )


def main() -> None:
    """Diagnose local data readiness without downloading datasets."""
    print_environment()
    print_dataset_status()
    print_recommendations()


if __name__ == "__main__":
    main()
