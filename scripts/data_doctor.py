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


def csv_image_count(path: Path) -> int | None:
    """Return unique image count for a manifest when possible."""
    if not path.exists():
        return None
    try:
        import pandas as pd

        df = pd.read_csv(path)
        if "image_id" in df.columns:
            return int(df["image_id"].nunique())
        if "image_path" in df.columns:
            return int(df["image_path"].nunique())
    except Exception:
        return None
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
    oxford_root = find_oxford_pet_root(ROOT / "data/raw/oxford_pet")

    checks = [
        ("Caption retrieval default manifest", ROOT / "data/processed/caption/caption_manifest.csv", status),
        ("Caption image features", ROOT / "outputs/features/caption_image_features.npy", status),
        ("Caption text features", ROOT / "outputs/features/caption_text_features.npy", status),
        ("Caption Top-K CSV", ROOT / "outputs/retrieval/caption_text_to_image_topk.csv", status),
        ("Caption retrieval metrics JSON", ROOT / "outputs/retrieval/caption_retrieval_metrics.json", status),
        ("Oxford Pet processed manifest", ROOT / "data/processed/oxford_pet/oxford_pet_segmentation_manifest.csv", found),
        ("Flickr30K manifest", ROOT / "data/processed/flickr30k/flickr30k_manifest.csv", found),
        ("DiffusionDB metadata manifest", ROOT / "data/processed/diffusiondb/diffusiondb_metadata_manifest.csv", found),
        ("DiffusionDB image quality manifest", ROOT / "data/processed/diffusiondb/diffusiondb_manifest.csv", found),
        ("Game asset manifest", ROOT / "data/processed/game_assets/game_asset_manifest.csv", found),
    ]

    print("\nDataset status:")
    print(f"- data/raw: {found(raw_root.exists())} ({rel(raw_root)})")
    print(f"- data/processed: {found(processed_root.exists())} ({rel(processed_root)})")
    print(f"- ModelScope cache dir: {found((ROOT / 'data/cache/modelscope').exists())} (data/cache/modelscope)")
    print(f"- HF cache dir: {found((ROOT / 'data/cache/huggingface').exists())} (data/cache/huggingface)")
    print(f"- Oxford Pet raw data: {found(oxford_root is not None)}")
    if oxford_root is not None:
        print(f"  Detected at: {rel(oxford_root)}")
    for name, path, formatter in checks:
        print(f"- {name}: {formatter(path.exists())} ({rel(path)})")
    caption_manifest = ROOT / "data/processed/caption/caption_manifest.csv"
    image_count = csv_image_count(caption_manifest)
    if image_count is not None:
        print(f"- Caption manifest image count: {image_count}")
        if image_count < 20:
            print("  WARNING: fewer than 20 images is too small for a useful retrieval demo.")
    print("- ModelScope COCO small slice: optional metadata/url sample, not the default retrieval dataset.")
    print("- DiffusionDB: metadata-only is for prompt analysis; local-images is for image quality evaluation.")


def print_recommendations() -> None:
    """Print Windows-friendly next commands."""
    print(
        "\nRecommended next commands:\n"
        "\n"
        "1. Recommended caption retrieval:\n"
        "   python scripts/prepare_caption_dataset.py --source hf --dataset lambda/naruto-blip-captions --max-samples 500 --force\n"
        "   python scripts/run_pipeline.py --task caption_retrieval\n"
        "\n"
        "2. Recommended CV segmentation:\n"
        "   python scripts/prepare_oxford_pet.py --source auto --max-samples 500 --make-pseudo-masks\n"
        "   python scripts/run_pipeline.py --task oxford_pet_segmentation\n"
        "\n"
        "3. Recommended local AIGC quality:\n"
        "   python scripts/prepare_local_game_assets.py --image-root <your_comfyui_outputs> --output data/processed/game_assets/game_asset_manifest.csv\n"
        "   python scripts/run_pipeline.py --task game_asset_quality\n"
        "\n"
        "4. DiffusionDB prompt metadata analysis:\n"
        "   python scripts/prepare_diffusiondb_subset.py --mode metadata-only --max-samples 1000\n"
        "   python scripts/analyze_prompts.py --manifest data/processed/diffusiondb/diffusiondb_metadata_manifest.csv --text-column prompt\n"
    )


def main() -> None:
    """Diagnose local data readiness without downloading datasets."""
    print_environment()
    print_dataset_status()
    print_recommendations()


if __name__ == "__main__":
    main()
