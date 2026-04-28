from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
from io import BytesIO
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

IMAGE_FIELDS = ["image", "image:FILE", "image_path", "img", "file", "filename", "filepath", "path", "url"]
CAPTION_FIELDS = ["caption", "captions", "text", "sentence", "sentences", "answer", "answer:Value", "description"]
ID_FIELDS = ["image_id", "id", "cocoid", "file_name", "filename"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def rel(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def safe_dataset_name(name: str) -> str:
    """Return a filesystem-safe dataset name."""
    return name.replace("\\", "__").replace("/", "__").replace(":", "_")


def normalize_modelscope_dataset_name(name: str) -> str:
    """Use a namespace/name dataset id for ModelScope CLI/Git URLs."""
    return name if "/" in name else f"modelscope/{name}"


def ensure_modelscope_cache(cache_root: Path) -> Path:
    """Create and configure project-local ModelScope cache directories."""
    cache = cache_root / "modelscope"
    for child in ["hub", "tmp", "locks"]:
        (cache / child).mkdir(parents=True, exist_ok=True)
    os.environ["MS_CACHE_HOME"] = str(cache.resolve())
    os.environ["MODELSCOPE_CACHE"] = str(cache.resolve())
    os.environ["MODELSCOPE_HOME"] = str(cache.resolve())
    return cache


def ensure_hf_cache(cache_root: Path) -> Path:
    """Create and configure project-local HuggingFace cache directories."""
    cache = cache_root / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache.resolve())
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    return cache


def infer_fields(columns_or_example: Any) -> dict[str, str | None]:
    """Infer image, caption and id fields from columns or a sample row."""
    if isinstance(columns_or_example, dict):
        columns = list(columns_or_example.keys())
    else:
        columns = list(columns_or_example)

    def pick(candidates: list[str]) -> str | None:
        lower = {str(c).lower(): str(c) for c in columns}
        for candidate in candidates:
            if candidate.lower() in lower:
                return lower[candidate.lower()]
        return None

    return {
        "image_field": pick(IMAGE_FIELDS),
        "caption_field": pick(CAPTION_FIELDS),
        "id_field": pick(ID_FIELDS),
    }


def normalize_captions(caption_obj: Any) -> list[str]:
    """Normalize caption-like values into a list of caption strings."""
    captions: list[str] = []

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            text = value.strip()
            if len(text) >= 3:
                captions.append(text)
            return
        if isinstance(value, dict):
            for key in ["text", "caption", "answer", "sentence", "raw", "description"]:
                if key in value:
                    add(value[key])
                    return
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add(item)
            return

    add(caption_obj)
    return list(dict.fromkeys(captions))


def _download_url(url: str, output_image_path: Path) -> bool:
    import requests

    urls = [url]
    if url.startswith("http://"):
        urls.append("https://" + url[len("http://"):])
    for candidate in urls:
        for verify in [True, False]:
            try:
                response = requests.get(candidate, timeout=20, verify=verify)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                output_image_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_image_path)
                return True
            except Exception as exc:
                mode = "verify=True" if verify else "verify=False fallback"
                print(f"Warning: failed to download image URL {candidate} ({mode}): {exc}")
    return False


def materialize_image(
    image_obj: Any,
    output_image_path: Path,
    source_root: Path | None = None,
    copy_images: bool = True,
) -> str | None:
    """Save or copy one image and return a path usable by downstream scripts."""
    if image_obj is None:
        return None
    if copy_images and output_image_path.exists():
        return rel(output_image_path)

    if isinstance(image_obj, dict):
        for key in ["path", "filename", "file", "filepath", "image_path", "url"]:
            if key in image_obj:
                return materialize_image(image_obj[key], output_image_path, source_root, copy_images)
        return None

    if hasattr(image_obj, "convert") and hasattr(image_obj, "save"):
        if not copy_images:
            return None
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        image_obj.convert("RGB").save(output_image_path)
        return rel(output_image_path)

    if isinstance(image_obj, np.ndarray):
        if not copy_images:
            return None
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(image_obj).convert("RGB").save(output_image_path)
        return rel(output_image_path)

    if isinstance(image_obj, (bytes, bytearray)):
        if not copy_images:
            return None
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.open(BytesIO(image_obj)).convert("RGB").save(output_image_path)
        return rel(output_image_path)

    value = str(image_obj).strip()
    if not value:
        return None
    if value.startswith(("http://", "https://")):
        if not copy_images:
            return value
        return rel(output_image_path) if _download_url(value, output_image_path) else None

    candidates = [Path(value)]
    if source_root is not None:
        candidates.append(source_root / value)
        candidates.append(source_root / Path(value).name)
        candidates.append(source_root / "images" / Path(value).name)
        candidates.append(source_root / "train2014" / Path(value).name)
        candidates.append(source_root / "val2014" / Path(value).name)
    for src in candidates:
        if src.exists() and src.is_file():
            if not copy_images:
                return rel(src)
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            suffix = src.suffix.lower() if src.suffix.lower() in IMAGE_EXTENSIONS else output_image_path.suffix
            dst = output_image_path.with_suffix(suffix)
            shutil.copyfile(src, dst)
            return rel(dst)
    name = Path(value).name
    if copy_images and name.startswith("COCO_"):
        split_dir = Path(value).parent.name or ("val2014" if "val" in name else "train2014")
        coco_url = f"https://images.cocodataset.org/{split_dir}/{name}"
        if _download_url(coco_url, output_image_path):
            return rel(output_image_path)
    return None


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        if isinstance(data.get("annotations"), list) and isinstance(data.get("images"), list):
            images = {str(img.get("id") or img.get("image_id")): img for img in data["images"] if isinstance(img, dict)}
            rows = []
            for ann in data["annotations"]:
                if not isinstance(ann, dict):
                    continue
                image_key = str(ann.get("image_id") or ann.get("id"))
                merged = {**images.get(image_key, {}), **ann}
                if "file_name" in merged and "image_path" not in merged:
                    merged["image_path"] = merged["file_name"]
                rows.append(merged)
            return rows
        for key in ["annotations", "images", "data", "rows", "records"]:
            if isinstance(data.get(key), list):
                return [x for x in data[key] if isinstance(x, dict)]
        return [data]
    return []


def _read_table(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path).to_dict("records")
    if suffix == ".json":
        return _read_json_records(path)
    if suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows
    if suffix == ".parquet":
        return pd.read_parquet(path).to_dict("records")
    return []


def _print_tree(root: Path, max_items: int = 40) -> None:
    print(f"Directory preview for {root}:")
    count = 0
    for path in sorted(root.rglob("*")):
        if count >= max_items:
            print("  ...")
            break
        depth = len(path.relative_to(root).parts)
        if depth <= 3:
            print(f"  {path.relative_to(root).as_posix()}")
            count += 1


def has_local_caption_files(root: Path) -> bool:
    """Return whether a local directory contains recognizable caption metadata."""
    if not root.exists():
        return False
    for pattern in ["*.csv", "*.json", "*.jsonl", "*.parquet"]:
        if any(root.rglob(pattern)):
            return True
    return False


def load_caption_records_from_local(root: Path, split: str | None = None) -> list[dict[str, Any]]:
    """Load caption rows from local CSV/JSON/JSONL/Parquet/Arrow-like folders."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Local root not found: {root}")

    patterns = ["*.csv", "*.json", "*.jsonl", "*.parquet"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.rglob(pattern)))

    for path in files:
        try:
            rows = _read_table(path)
        except Exception as exc:
            print(f"Warning: failed to read {path}: {exc}")
            continue
        if not rows:
            continue
        if split and any("split" in row for row in rows):
            split_rows = [row for row in rows if str(row.get("split")) == split]
            if split_rows:
                rows = split_rows
        for row in rows:
            row.setdefault("_source_root", str(path.parent))
        print(f"Loaded {len(rows)} local records from {rel(path)}")
        return rows

    try:
        from datasets import load_from_disk

        dataset = load_from_disk(str(root))
        if hasattr(dataset, "keys"):
            key = split if split in dataset else next(iter(dataset.keys()))
            records = [dict(x) for x in dataset[key]]
        else:
            records = [dict(x) for x in dataset]
        for row in records:
            row.setdefault("_source_root", str(root))
        return records
    except Exception:
        pass

    _print_tree(root)
    raise FileNotFoundError(
        "Cannot find recognizable local caption files. Expected CSV, JSON, JSONL, "
        "Parquet, or a datasets.load_from_disk directory."
    )


def _try_modelscope_sdk(dataset_name: str, split: str, cache: Path):
    from modelscope.msdatasets import MsDataset

    errors = []
    for candidate in [split, "train", "validation", "test"]:
        try:
            return MsDataset.load(dataset_name, split=candidate, cache_dir=str(cache)), candidate
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def _try_modelscope_cli(dataset_name: str, local_dir: Path) -> bool:
    local_dir.mkdir(parents=True, exist_ok=True)
    cli_dataset_name = normalize_modelscope_dataset_name(dataset_name)
    commands = [
        [sys.executable, "-m", "modelscope.cli.cli", "download", "--dataset", cli_dataset_name, "--local_dir", str(local_dir)],
        ["modelscope", "download", "--dataset", cli_dataset_name, "--local_dir", str(local_dir)],
    ]
    for cmd in commands:
        try:
            print("$ " + " ".join(cmd))
            subprocess.run(cmd, cwd=ROOT, check=True)
            return True
        except Exception as exc:
            print(f"Warning: ModelScope CLI command failed: {exc}")
    return False


def load_modelscope_dataset_safe(
    dataset_name: str,
    split: str,
    cache_root: Path,
    method: str = "auto",
    local_dir: Path | None = None,
):
    """Load a ModelScope dataset with project-local cache and CLI fallback."""
    cache = ensure_modelscope_cache(cache_root)
    local_dir = local_dir or ROOT / "data/raw/modelscope" / safe_dataset_name(dataset_name)
    if has_local_caption_files(local_dir):
        print(f"Found local ModelScope dataset files: {rel(local_dir)}")
        return load_caption_records_from_local(local_dir, split), split

    if method in {"auto", "sdk"}:
        try:
            return _try_modelscope_sdk(dataset_name, split, cache)
        except Exception as exc:
            print("ModelScope SDK load failed.")
            print(f"Reason: {type(exc).__name__}: {exc}")
            print(f"Using project cache: {rel(cache)}")
            if method == "sdk":
                raise

    if method in {"auto", "cli"}:
        if _try_modelscope_cli(dataset_name, local_dir):
            return load_caption_records_from_local(local_dir, split), split

    print(
        "ModelScope SDK/CLI did not finish. You can try manual Git download, for example:\n"
        f"git clone https://www.modelscope.cn/datasets/{normalize_modelscope_dataset_name(dataset_name)}.git {local_dir.as_posix()}\n"
        f"python scripts/prepare_caption_dataset.py --source local --local-root {rel(local_dir)} --max-samples 500"
    )
    if local_dir.exists():
        return load_caption_records_from_local(local_dir, split), split
    raise RuntimeError("Failed to load ModelScope dataset by SDK, CLI, or local fallback.")


def load_hf_dataset_safe(dataset_name: str, split: str, cache_root: Path, token: str | None = None):
    """Load a HuggingFace dataset with project-local cache and legacy-script diagnostics."""
    hf_cache = ensure_hf_cache(cache_root)
    try:
        from datasets import get_dataset_split_names, load_dataset
    except ImportError as exc:
        raise ImportError("Please install datasets for HuggingFace loading: pip install datasets") from exc

    candidates = [split, "train", "validation", "test"]
    try:
        split_names = get_dataset_split_names(dataset_name, token=token)
        candidates.extend([s for s in split_names if s not in candidates])
    except Exception:
        pass

    errors = []
    for candidate in candidates:
        try:
            print(
                f"Loading HuggingFace dataset {dataset_name} split={candidate} "
                f"with cache {rel(hf_cache)} ..."
            )
            return load_dataset(dataset_name, split=candidate, token=token, cache_dir=str(hf_cache)), candidate
        except Exception as exc:
            message = str(exc)
            if "Dataset scripts are no longer supported" in message:
                local_dir = ROOT / "data/raw/hf" / safe_dataset_name(dataset_name)
                try:
                    from huggingface_hub import snapshot_download

                    print("HF load_dataset found a legacy script; trying snapshot_download + local file scan.")
                    snapshot_download(
                        repo_id=dataset_name,
                        repo_type="dataset",
                        local_dir=str(local_dir),
                        token=token,
                    )
                    if has_local_caption_files(local_dir):
                        return load_caption_records_from_local(local_dir, split), split
                except Exception as fallback_exc:
                    print(f"HF snapshot/local fallback failed: {type(fallback_exc).__name__}: {fallback_exc}")
                print(
                    "\nThis dataset uses an old HuggingFace dataset script.\n"
                    "datasets>=4.0 no longer supports loading dataset scripts.\n"
                    "This is not an HF token/login problem.\n"
                    "Please use one of:\n"
                    "A) --dataset HuggingFaceM4/COCO\n"
                    "B) --source modelscope --dataset coco_captions_small_slice\n"
                    "C) --source local --local-root data/raw/caption_dataset\n"
                    "D) create a separate legacy env with datasets<4.0 only if you really need this dataset.\n"
                )
                raise RuntimeError(message) from None
            if any(text in message.lower() for text in ["gated", "private", "401", "403", "rate limit"]):
                print("This may be a gated/private/rate-limited dataset. Use --hf-token if you have access.")
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Failed to load HuggingFace dataset. If this is lambda/naruto-blip-captions, "
        "it is a standard Parquet dataset, not a legacy script; check HF network/cache, "
        "try a smaller --max-samples, or use --source local with an already downloaded Parquet directory. "
        + "; ".join(errors)
    )


def _as_records(dataset: Any) -> list[dict[str, Any]]:
    if isinstance(dataset, list):
        return [dict(x) for x in dataset]
    return [dict(x) for x in dataset]


def build_manifest_from_records(records: list[dict[str, Any]], args: argparse.Namespace, actual_split: str) -> pd.DataFrame:
    """Convert raw dataset records to the unified caption manifest."""
    output_root = Path(args.output_root)
    image_dir = output_root / "images"
    rows = []
    skipped = 0
    image_index = 0

    if not records:
        raise ValueError("No records loaded from caption dataset.")

    field_info = infer_fields(records[0].keys())
    image_field = args.image_field or field_info["image_field"]
    caption_field = args.caption_field or field_info["caption_field"]
    id_field = args.id_field or field_info["id_field"]
    if not image_field or not caption_field:
        raise ValueError(
            "Cannot infer image/caption fields.\n"
            f"Available columns: {list(records[0].keys())}\n"
            "Please specify field mapping or update FIELD_CANDIDATES."
        )

    seen_images: dict[str, str] = {}
    for raw_idx, row in enumerate(tqdm(records, desc="prepare caption manifest")):
        if image_index >= args.max_samples:
            break
        captions = normalize_captions(row.get(caption_field))
        if not captions:
            skipped += 1
            continue
        raw_image_value = row.get(image_field)
        image_id_source = row.get(id_field) or row.get("image_id") or row.get("id")
        if image_id_source is None and isinstance(raw_image_value, str):
            image_id_source = raw_image_value
        image_id = str(image_id_source if image_id_source is not None else raw_idx)
        output_image_path = image_dir / f"{image_index:06d}.jpg"
        source_root = Path(row["_source_root"]) if row.get("_source_root") else None
        image_path = seen_images.get(image_id)
        if image_path is None:
            image_path = materialize_image(
                row.get(image_field),
                output_image_path,
                source_root=source_root,
                copy_images=not args.no_copy_images,
            )
            if image_path is None:
                skipped += 1
                continue
            seen_images[image_id] = image_path
            image_index += 1

        for local_caption_id, caption in enumerate(captions):
            unique_caption_id = f"{image_id}_{len(rows)}_{local_caption_id}"
            rows.append({
                "image_path": image_path,
                "text": caption,
                "label": "caption",
                "split": actual_split,
                "source": args.source,
                "image_id": image_id,
                "caption_id": unique_caption_id,
                "dataset_name": args.dataset if args.source != "local" else rel(Path(args.local_root)),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(
            "No caption pairs were written. Images may be missing or blocked by network/SSL. "
            "Use --source local with image files present, or retry when image URLs are reachable."
        )
    output_path = Path(args.manifest_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved manifest: {rel(output_path)}")
    num_images = int(df["image_id"].nunique() if not df.empty else 0)
    print(f"Images: {num_images}")
    print(f"Caption pairs: {len(df)}")
    print(f"Skipped raw samples: {skipped}")
    print(f"Dataset: {args.dataset if args.source != 'local' else rel(Path(args.local_root))}")
    if args.source == "modelscope" and num_images < 20:
        print(
            f"Warning: this dataset produced only {num_images} materialized images.\n"
            "modelscope/coco_captions_small_slice is a tiny metadata/url-based sample and is not suitable for a meaningful retrieval benchmark.\n"
            "Recommended default:\n"
            "python scripts/prepare_caption_dataset.py --source hf --dataset lambda/naruto-blip-captions --max-samples 500 --force"
        )
    return df


def inspect_records(records: list[dict[str, Any]], max_rows: int = 3) -> None:
    """Print dataset structure for inspection."""
    print(f"Loaded records: {len(records)}")
    if not records:
        return
    print(f"Columns: {list(records[0].keys())}")
    print(f"Inferred fields: {infer_fields(records[0].keys())}")
    for idx, row in enumerate(records[:max_rows]):
        preview = {k: str(v)[:120] for k, v in row.items() if not k.startswith("_")}
        print(f"Row {idx}: {preview}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a robust image-caption manifest from ModelScope, HF, or local files.")
    parser.add_argument("--source", default="modelscope", choices=["modelscope", "hf", "local"])
    parser.add_argument("--dataset", default="lambda/naruto-blip-captions")
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--output-root", default="data/processed/caption")
    parser.add_argument("--manifest-output", default="data/processed/caption/caption_manifest.csv")
    parser.add_argument("--cache-root", default="data/cache")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--no-copy-images", action="store_true")
    parser.add_argument("--hf-token")
    parser.add_argument("--hf-local-dir")
    parser.add_argument("--modelscope-cache-dir")
    parser.add_argument("--modelscope-local-dir")
    parser.add_argument("--modelscope-method", default="auto", choices=["sdk", "cli", "git", "auto"])
    parser.add_argument("--local-root", default="data/raw/caption_dataset")
    parser.add_argument("--image-field")
    parser.add_argument("--caption-field")
    parser.add_argument("--id-field")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = Path(args.manifest_output)
    if manifest.exists() and not args.force and not args.inspect_only:
        print(f"Caption manifest already exists: {rel(manifest)}")
        print("Use --force to regenerate it.")
        return

    cache_root = Path(args.cache_root)
    if args.source == "local" or args.hf_local_dir:
        records = load_caption_records_from_local(Path(args.hf_local_dir or args.local_root), args.split)
        actual_split = args.split
    elif args.source == "modelscope":
        ms_cache_root = Path(args.modelscope_cache_dir).parent if args.modelscope_cache_dir else cache_root
        local_dir = Path(args.modelscope_local_dir) if args.modelscope_local_dir else None
        dataset, actual_split = load_modelscope_dataset_safe(
            args.dataset,
            args.split,
            ms_cache_root,
            method=args.modelscope_method,
            local_dir=local_dir,
        )
        records = _as_records(dataset)
    else:
        dataset, actual_split = load_hf_dataset_safe(args.dataset, args.split, cache_root, token=args.hf_token)
        records = _as_records(dataset)

    records = records[: args.max_samples]
    if args.inspect_only:
        inspect_records(records)
        return
    build_manifest_from_records(records, args, actual_split)


if __name__ == "__main__":
    main()
