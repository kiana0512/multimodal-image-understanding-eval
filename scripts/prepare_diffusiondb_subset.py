from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import json
import os
from shutil import copyfile

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def rel(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def infer_asset_type(prompt: str) -> str:
    """Infer a coarse asset type from prompt keywords."""
    text = prompt.lower()
    if any(k in text for k in ["icon", "logo", "ui", "button"]):
        return "ui_icon"
    if any(k in text for k in ["character", "girl", "boy", "person", "warrior"]):
        return "character"
    if any(k in text for k in ["landscape", "scene", "city", "room"]):
        return "scene"
    return "general"


def load_table(path: Path) -> pd.DataFrame:
    """Load metadata from CSV/JSON/JSONL/Parquet."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return pd.DataFrame(rows)
    raise ValueError(f"Unsupported metadata file: {path}")


def find_metadata_file(root: Path) -> Path | None:
    """Find a local metadata file."""
    for name in ["metadata.parquet", "metadata.csv", "metadata.jsonl", "metadata.json"]:
        path = root / name
        if path.exists():
            return path
    for pattern in ["*.parquet", "*.csv", "*.jsonl", "*.json"]:
        matches = sorted(root.rglob(pattern))
        if matches:
            return matches[0]
    return None


def try_download_metadata(raw_root: Path) -> Path | None:
    """Try to download only DiffusionDB metadata without loading image dataset scripts."""
    raw_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str((ROOT / "data/cache/huggingface").resolve()))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        return None
    candidates = [
        "metadata.parquet",
        "2m_first_1k/metadata.parquet",
        "metadata-large.parquet",
    ]
    for filename in candidates:
        try:
            print(f"Trying to download DiffusionDB metadata only: {filename}")
            downloaded = hf_hub_download(
                repo_id="poloclub/diffusiondb",
                repo_type="dataset",
                filename=filename,
                local_dir=str(raw_root),
            )
            return Path(downloaded)
        except Exception as exc:
            print(f"Warning: failed to download DiffusionDB metadata file {filename}: {exc}")
    return None


def prepare_metadata_only(args: argparse.Namespace) -> None:
    """Prepare prompt-only DiffusionDB metadata manifest."""
    raw_root = Path(args.local_root)
    metadata_file = find_metadata_file(raw_root) or try_download_metadata(raw_root)
    if metadata_file is None:
        raise FileNotFoundError(
            "DiffusionDB metadata.parquet was not found and could not be downloaded.\n"
            "Manual option: download a DiffusionDB metadata parquet/csv/json file into data/raw/diffusiondb,\n"
            "then run: python scripts/prepare_diffusiondb_subset.py --mode metadata-only --max-samples 1000"
        )
    df = load_table(metadata_file).head(args.max_samples).copy()
    out = pd.DataFrame({
        "prompt": df.get("prompt", df.get("text", "")),
        "seed": df.get("seed", ""),
        "cfg": df.get("cfg", df.get("guidance_scale", "")),
        "steps": df.get("steps", df.get("num_inference_steps", "")),
        "sampler": df.get("sampler", ""),
        "width": df.get("width", ""),
        "height": df.get("height", ""),
        "image_nsfw": df.get("image_nsfw", ""),
        "prompt_nsfw": df.get("prompt_nsfw", ""),
        "source": f"diffusiondb_metadata:{rel(metadata_file)}",
    })
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "diffusiondb_metadata_manifest.csv"
    out.to_csv(manifest, index=False)
    print(f"Saved metadata manifest: {rel(manifest)}")
    print(f"Rows: {len(out)}")
    print("Status: prompt metadata only; no images were downloaded.")


def prepare_local_images(args: argparse.Namespace) -> None:
    """Build an AIGC quality manifest from local DiffusionDB images and metadata."""
    root = Path(args.local_root)
    if not root.exists():
        raise FileNotFoundError(f"Local DiffusionDB root not found: {root}")
    metadata_file = find_metadata_file(root)
    meta = load_table(metadata_file) if metadata_file else pd.DataFrame()
    images = sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    rows = []
    image_out = Path(args.output_dir) / "images"
    for idx, image in enumerate(images[: args.max_samples]):
        meta_row = meta.iloc[idx].to_dict() if idx < len(meta) else {}
        prompt = str(meta_row.get("prompt") or meta_row.get("text") or "")
        dst = image_out / f"diffusiondb_{idx:06d}{image.suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        copyfile(image, dst)
        rows.append({
            "image_path": rel(dst),
            "prompt": prompt,
            "negative_prompt": meta_row.get("negative_prompt", ""),
            "style_tag": meta_row.get("style", "unknown") or "unknown",
            "asset_type": infer_asset_type(prompt),
            "seed": meta_row.get("seed", ""),
            "model_name": meta_row.get("model_name", meta_row.get("model", "unknown")) or "unknown",
            "split": "sample",
            "source": f"local_diffusiondb:{rel(root)}",
        })
    if not rows:
        raise ValueError("No local DiffusionDB images found.")
    manifest = Path(args.output_dir) / "diffusiondb_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Saved local image manifest: {rel(manifest)}")
    print(f"Rows: {len(rows)}")


def prepare_legacy_loader(args: argparse.Namespace) -> None:
    """Legacy DiffusionDB image loader. Use only when explicitly requested."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Please install datasets to use legacy-loader mode.") from exc
    try:
        dataset = load_dataset("poloclub/diffusiondb", args.config_name, split="train", streaming=True)
        first = next(iter(dataset))
        print(f"Legacy loader can access dataset. First row keys: {list(dict(first).keys())}")
    except Exception as exc:
        if "Dataset scripts are no longer supported" in str(exc):
            raise RuntimeError(
                "poloclub/diffusiondb uses an old HuggingFace dataset script. datasets>=4.0 no longer supports it.\n"
                "Do not downgrade the main llm environment. Use --mode metadata-only, --mode local-images,\n"
                "or create a separate legacy environment only if you really need this loader."
            ) from None
        raise


def main() -> None:
    """Prepare DiffusionDB as metadata-only by default."""
    parser = argparse.ArgumentParser(description="Prepare DiffusionDB metadata or local image subsets.")
    parser.add_argument("--mode", default="metadata-only", choices=["metadata-only", "local-images", "legacy-loader"])
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--output-dir", default="data/processed/diffusiondb")
    parser.add_argument("--local-root", default="data/raw/diffusiondb")
    parser.add_argument("--config-name", default="2m_first_1k")
    args = parser.parse_args()
    if args.mode == "metadata-only":
        prepare_metadata_only(args)
    elif args.mode == "local-images":
        prepare_local_images(args)
    else:
        prepare_legacy_loader(args)


if __name__ == "__main__":
    main()
