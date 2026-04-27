from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import pandas as pd
import numpy as np
from mm_eval.utils.config import load_config
from mm_eval.models.clip_encoder import ClipEncoder


def main() -> None:
    """Extract image and text CLIP/OpenCLIP features."""
    parser = argparse.ArgumentParser(description="Extract CLIP/OpenCLIP features from a manifest.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(cfg["manifest_path"])
    out = Path(cfg.get("feature_dir", cfg.get("output_dir", "outputs/features")))
    out.mkdir(parents=True, exist_ok=True)
    encoder = ClipEncoder(model_name=cfg.get("model_name", "ViT-B-32"), device=cfg.get("device"))
    image_features = encoder.encode_images(
        df[cfg.get("image_column", "image_path")].astype(str).tolist(),
        batch_size=int(cfg.get("batch_size", 32)),
    )
    text_features = encoder.encode_texts(
        df[cfg.get("text_column", "text")].astype(str).tolist(),
        batch_size=int(cfg.get("batch_size", 32)),
    )
    np.save(out / cfg.get("image_feature_name", "image_features.npy"), image_features)
    np.save(out / cfg.get("text_feature_name", "text_features.npy"), text_features)
    print(f"Saved features to {out}")


if __name__ == "__main__":
    main()
