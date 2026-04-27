from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm


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


def _save_image(image_obj, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(image_obj, "save"):
        image_obj.convert("RGB").save(output_path)
        return True
    src = Path(str(image_obj))
    if src.exists():
        from shutil import copyfile
        copyfile(src, output_path)
        return True
    return False


def main() -> None:
    """Prepare a tiny DiffusionDB subset for AIGC quality evaluation."""
    parser = argparse.ArgumentParser(description="Prepare a small DiffusionDB subset with streaming by default.")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--output-dir", default="data/processed/diffusiondb")
    parser.add_argument("--config-name", default="2m_first_1k", help="DiffusionDB config name; keep small by default.")
    parser.add_argument("--no-streaming", action="store_true")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Please install datasets: `pip install datasets` or create the `llm` conda env.") from exc

    try:
        dataset = load_dataset("poloclub/diffusiondb", args.config_name, split="train", streaming=not args.no_streaming)
    except Exception:
        try:
            dataset = load_dataset("poloclub/diffusiondb", split="train", streaming=not args.no_streaming)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DiffusionDB. It is a very large dataset; use streaming, HF mirror settings, "
                "or manually prepare data/processed/diffusiondb/diffusiondb_manifest.csv."
            ) from exc

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    rows = []
    for idx, row in enumerate(tqdm(dataset, total=args.max_samples, desc="prepare diffusiondb")):
        if idx >= args.max_samples:
            break
        row_dict = dict(row)
        image = row_dict.get("image")
        prompt = str(row_dict.get("prompt") or row_dict.get("text") or "")
        if image is None or not prompt:
            continue
        image_path = image_dir / f"diffusiondb_{idx:06d}.jpg"
        if not _save_image(image, image_path):
            continue
        rows.append({
            "image_path": image_path.as_posix(),
            "prompt": prompt,
            "negative_prompt": row_dict.get("negative_prompt", ""),
            "style_tag": row_dict.get("style", "unknown") or "unknown",
            "asset_type": infer_asset_type(prompt),
            "seed": row_dict.get("seed", ""),
            "model_name": row_dict.get("model_name", row_dict.get("model", "unknown")) or "unknown",
            "split": "sample",
            "source": "hf:poloclub/diffusiondb",
        })
    manifest = output_dir / "diffusiondb_manifest.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"Saved {len(rows)} rows to {manifest}. This is a small experimental subset, not the full DiffusionDB.")


if __name__ == "__main__":
    main()
