from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def _extract_text(row: dict) -> str:
    for key in ["caption", "text", "sentence", "sentences"]:
        value = row.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            return str(value[0])
    return ""


def _save_image(image_obj, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image_obj, "save"):
        image_obj.convert("RGB").save(output_path)
        return
    src = Path(str(image_obj))
    if src.exists():
        from shutil import copyfile
        copyfile(src, output_path)
        return
    raise ValueError("Cannot save image from this row. Use manual data placement if ModelScope schema changed.")


def main() -> None:
    """Prepare COCO Caption subset from ModelScope when available."""
    parser = argparse.ArgumentParser(description="Prepare COCO caption subset from ModelScope.")
    parser.add_argument("--dataset", default="modelscope/coco_captions_small_slice")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="data/processed/coco_caption")
    args = parser.parse_args()

    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:
        raise ImportError("Please install modelscope or create the `llm` conda env.") from exc

    try:
        ds = MsDataset.load(args.dataset, split=args.split)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load COCO Caption from ModelScope. This dataset is a backup flow. "
            "You can manually place images under data/processed/coco_caption/images and create coco_caption_manifest.csv."
        ) from exc

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    rows = []
    for idx, row in enumerate(tqdm(ds, desc="prepare coco caption")):
        if idx >= args.max_samples:
            break
        row_dict = dict(row)
        image = row_dict.get("image") or row_dict.get("img") or row_dict.get("image_path")
        text = _extract_text(row_dict)
        if image is None or not text:
            continue
        image_id = str(row_dict.get("image_id") or row_dict.get("id") or idx)
        image_path = image_dir / f"{image_id}.jpg"
        _save_image(image, image_path)
        rows.append({
            "image_path": image_path.as_posix(),
            "text": text,
            "label": "image_caption",
            "split": args.split,
            "source": f"modelscope:{args.dataset}",
            "image_id": image_id,
            "caption_id": row_dict.get("caption_id", 0),
        })
    manifest = output_dir / "coco_caption_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Saved {len(rows)} rows to {manifest}")


if __name__ == "__main__":
    main()
