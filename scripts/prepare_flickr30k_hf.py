from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def _captions_from_row(row: dict) -> list[str]:
    for key in ["caption", "captions", "sentences", "text"]:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            captions = []
            for item in value:
                if isinstance(item, str):
                    captions.append(item)
                elif isinstance(item, dict):
                    captions.append(str(item.get("raw") or item.get("text") or item.get("caption") or item))
            return captions
    return []


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
    raise ValueError("Row image field is neither a PIL image nor an existing path.")


def main() -> None:
    """Prepare Flickr30K image-caption manifest from HuggingFace."""
    parser = argparse.ArgumentParser(description="Prepare nlphuji/flickr30k subset from HuggingFace datasets.")
    parser.add_argument("--split", default="test", choices=["train", "val", "validation", "test"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="data/processed/flickr30k")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Please install datasets: `pip install datasets` or create the `llm` conda env.") from exc

    split = "validation" if args.split == "val" else args.split
    try:
        dataset = load_dataset("nlphuji/flickr30k", split=split)
    except Exception as exc:
        raise RuntimeError(
            "Failed to download/load nlphuji/flickr30k from HuggingFace. "
            "Check network/SSL/HF mirror settings, or prepare data manually as described in docs/how_to_run_real_data_zh.md."
        ) from exc

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    rows = []
    for idx, row in enumerate(tqdm(dataset, desc="prepare flickr30k")):
        if idx >= args.max_samples:
            break
        row_dict = dict(row)
        image = row_dict.get("image") or row_dict.get("img")
        if image is None:
            raise ValueError(f"Cannot find image field in row keys: {list(row_dict.keys())}")
        image_id = str(row_dict.get("image_id") or row_dict.get("img_id") or row_dict.get("filename") or idx)
        image_path = image_dir / f"{image_id}.jpg"
        _save_image(image, image_path)
        captions = _captions_from_row(row_dict)
        if not captions:
            raise ValueError(f"Cannot find caption field in row keys: {list(row_dict.keys())}")
        for caption_id, caption in enumerate(captions):
            rows.append({
                "image_path": image_path.as_posix(),
                "text": caption,
                "label": "image_caption",
                "split": args.split,
                "source": "hf:nlphuji/flickr30k",
                "image_id": image_id,
                "caption_id": caption_id,
            })
    manifest = output_dir / "flickr30k_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Saved {len(rows)} image-caption rows to {manifest}")


if __name__ == "__main__":
    main()
