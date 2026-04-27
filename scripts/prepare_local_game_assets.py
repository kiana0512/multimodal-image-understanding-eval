from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path
import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def infer_asset_type(path: Path) -> str:
    """Infer asset type from path text."""
    text = path.as_posix().lower()
    if any(k in text for k in ["icon", "ui", "button"]):
        return "ui_icon"
    if any(k in text for k in ["character", "char", "role"]):
        return "character"
    if any(k in text for k in ["scene", "landscape", "city", "room"]):
        return "scene"
    return "general"


def main() -> None:
    """Prepare a local game asset manifest from ComfyUI/SD output folders."""
    parser = argparse.ArgumentParser(description="Scan local game asset images and create/merge a manifest.")
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--prompts-csv", help="Optional prompts.csv with image_path or filename plus prompt metadata.")
    parser.add_argument("--output", default="data/processed/game_assets/game_asset_manifest.csv")
    args = parser.parse_args()

    image_root = Path(args.image_root)
    if not image_root.exists():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    images = sorted([p for p in image_root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not images:
        raise ValueError(f"No images found under {image_root}")

    prompt_df = None
    if args.prompts_csv:
        prompt_df = pd.read_csv(args.prompts_csv)

    rows = []
    for idx, image_path in enumerate(images):
        rel_or_abs = image_path.as_posix()
        prompt_meta = {}
        if prompt_df is not None:
            if "image_path" in prompt_df.columns:
                matched = prompt_df[prompt_df["image_path"].astype(str).map(lambda v: Path(v).name) == image_path.name]
            elif "filename" in prompt_df.columns:
                matched = prompt_df[prompt_df["filename"].astype(str) == image_path.name]
            else:
                matched = pd.DataFrame()
            if not matched.empty:
                prompt_meta = matched.iloc[0].to_dict()
        rows.append({
            "image_path": rel_or_abs,
            "asset_id": prompt_meta.get("asset_id", image_path.stem),
            "asset_type": prompt_meta.get("asset_type", infer_asset_type(image_path)),
            "style_tag": prompt_meta.get("style_tag", "unknown"),
            "prompt": prompt_meta.get("prompt", ""),
            "negative_prompt": prompt_meta.get("negative_prompt", ""),
            "seed": prompt_meta.get("seed", ""),
            "model_name": prompt_meta.get("model_name", "unknown"),
            "split": prompt_meta.get("split", "val"),
            "source": prompt_meta.get("source", "local_comfyui"),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved {len(rows)} local game asset rows to {output}")
    if prompt_df is None:
        print("No prompts.csv provided. Please fill prompt/style/seed fields before formal evaluation.")


if __name__ == "__main__":
    main()
