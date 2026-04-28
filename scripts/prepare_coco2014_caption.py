from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def rel(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def split_names(split: str) -> list[str]:
    """Expand trainval to concrete COCO split names."""
    if split == "trainval":
        return ["train", "val"]
    return [split]


def annotation_path(root: Path, split: str) -> Path:
    """Return the official COCO captions JSON path for a split."""
    return root / "annotations" / f"captions_{split}2014.json"


def image_dir(root: Path, split: str) -> Path:
    """Return the official COCO image directory for a split."""
    return root / f"{split}2014"


def missing_data_message(root: Path, split: str) -> str:
    """Build actionable missing-data instructions."""
    return (
        f"COCO2014 {split} files are missing under {rel(root)}.\n\n"
        "Run the official downloader first, for example:\n"
        f"python scripts/download_coco2014_official.py --split {split} --download --extract"
    )


def load_split_rows(root: Path, split: str) -> tuple[list[dict[str, object]], int, int]:
    """Load one COCO split and return manifest rows plus image/missing counts."""
    ann_path = annotation_path(root, split)
    img_dir = image_dir(root, split)
    if not ann_path.exists() or not img_dir.exists():
        raise FileNotFoundError(missing_data_message(root, split))

    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in data.get("images", [])}
    annotations = data.get("annotations", [])
    rows: list[dict[str, object]] = []
    missing_images: set[int] = set()
    existing_images: set[int] = set()

    for ann in annotations:
        original_image_id = int(ann["image_id"])
        image = images.get(original_image_id)
        if image is None:
            missing_images.add(original_image_id)
            continue
        path = img_dir / str(image["file_name"])
        if not path.exists():
            missing_images.add(original_image_id)
            continue
        existing_images.add(original_image_id)
        rows.append({
            "image_path": rel(path),
            "text": str(ann.get("caption", "")).strip(),
            "label": "caption",
            "split": split,
            "source": "official_coco2014",
            "image_id": f"{split}_{original_image_id}",
            "caption_id": f"{split}_{ann['id']}",
            "dataset_name": "coco2014_caption",
        })

    return rows, len(existing_images), len(missing_images)


def limit_rows(
    rows: list[dict[str, object]],
    max_samples: int,
    max_images: int | None,
    seed: int,
) -> list[dict[str, object]]:
    """Shuffle and limit rows by image count and caption-pair count."""
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)

    if max_images is not None:
        selected_images: set[str] = set()
        image_limited: list[dict[str, object]] = []
        for row in rows:
            image_id = str(row["image_id"])
            if image_id not in selected_images:
                if len(selected_images) >= max_images:
                    continue
                selected_images.add(image_id)
            image_limited.append(row)
        rows = image_limited

    return rows[:max_samples]


def maybe_copy_images(rows: list[dict[str, object]], output_root: Path) -> list[dict[str, object]]:
    """Copy selected COCO images into processed caption images when requested."""
    copied: dict[str, str] = {}
    image_out = output_root / "images"
    new_rows: list[dict[str, object]] = []
    for row in rows:
        old_path = Path(str(row["image_path"]))
        if not old_path.is_absolute():
            old_path = ROOT / old_path
        key = str(row["image_id"])
        if key not in copied:
            dst = image_out / old_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(old_path, dst)
            copied[key] = rel(dst)
        copied_row = dict(row)
        copied_row["image_path"] = copied[key]
        new_rows.append(copied_row)
    return new_rows


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build a caption retrieval manifest from official COCO2014 files.")
    parser.add_argument("--root", default="data/raw/coco2014")
    parser.add_argument("--split", default="val", choices=["train", "val", "trainval"])
    parser.add_argument("--output-root", default="data/processed/caption")
    parser.add_argument("--manifest-output", default="data/processed/caption/caption_manifest.csv")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--copy-images", dest="copy_images", action="store_true")
    parser.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    parser.set_defaults(copy_images=False)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Build the official COCO2014 caption retrieval manifest."""
    args = parse_args()
    manifest = Path(args.manifest_output)
    if manifest.exists() and not args.force:
        print(f"Caption manifest already exists: {rel(manifest)}")
        print("Use --force to regenerate it.")
        return

    root = Path(args.root)
    all_rows: list[dict[str, object]] = []
    total_images = 0
    total_missing = 0
    for concrete_split in split_names(args.split):
        rows, image_count, missing_count = load_split_rows(root, concrete_split)
        all_rows.extend(rows)
        total_images += image_count
        total_missing += missing_count

    rows = limit_rows(all_rows, args.max_samples, args.max_images, args.seed)
    if args.copy_images:
        rows = maybe_copy_images(rows, Path(args.output_root))
    if not rows:
        raise RuntimeError("No COCO caption pairs were written. Check images and annotations.")

    columns = ["image_path", "text", "label", "split", "source", "image_id", "caption_id", "dataset_name"]
    df = pd.DataFrame(rows, columns=columns)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest, index=False)

    print(f"Saved manifest: {rel(manifest)}")
    print(f"Images: {df['image_id'].nunique()}")
    print(f"Caption pairs: {len(df)}")
    print(f"Missing images: {total_missing}")
    print(f"Split: {args.split}")
    print("Dataset: official_coco2014")
    if args.max_images is not None:
        print(f"Max images requested: {args.max_images}")
    print(f"Source images detected before limiting: {total_images}")


if __name__ == "__main__":
    main()
