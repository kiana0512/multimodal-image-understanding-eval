from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import pandas as pd
import numpy as np
from mm_eval.utils.config import load_config
from mm_eval.models.clip_encoder import ClipEncoder


def _feature_path(cfg: dict, explicit_key: str, legacy_name_key: str, default_name: str) -> Path:
    """Resolve feature output path from explicit path or legacy feature_dir/name keys."""
    if cfg.get(explicit_key):
        return Path(cfg[explicit_key])
    return Path(cfg.get("feature_dir", cfg.get("output_dir", "outputs/features"))) / cfg.get(
        legacy_name_key,
        default_name,
    )


def main() -> None:
    """Extract image and text CLIP/OpenCLIP features."""
    parser = argparse.ArgumentParser(description="Extract CLIP/OpenCLIP features from a manifest.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(cfg["manifest_path"])
    image_col = cfg.get("image_column", "image_path")
    text_col = cfg.get("text_column", "text")
    image_id_col = cfg.get("image_id_column")
    caption_id_col = cfg.get("caption_id_column")
    image_feature_path = _feature_path(cfg, "image_feature_path", "image_feature_name", "image_features.npy")
    text_feature_path = _feature_path(cfg, "text_feature_path", "text_feature_name", "text_features.npy")
    image_feature_path.parent.mkdir(parents=True, exist_ok=True)
    text_feature_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = ClipEncoder(
        model_name=cfg.get("model_name", "ViT-B-32"),
        pretrained=cfg.get("pretrained", "laion2b_s34b_b79k"),
        device=cfg.get("device"),
    )
    if image_id_col and image_id_col in df.columns:
        image_df = df.drop_duplicates(subset=[image_id_col]).reset_index(drop=True)
    else:
        image_df = df.reset_index(drop=True)
    image_features = encoder.encode_images(
        image_df[image_col].astype(str).tolist(),
        batch_size=int(cfg.get("batch_size", 32)),
    )
    text_features = encoder.encode_texts(
        df[text_col].astype(str).tolist(),
        batch_size=int(cfg.get("batch_size", 32)),
    )
    np.save(image_feature_path, image_features)
    np.save(text_feature_path, text_features)

    metadata_path = cfg.get("metadata_path")
    if metadata_path:
        metadata = df.copy()
        metadata.to_csv(metadata_path, index=False)
        gallery_path = Path(metadata_path).with_name(Path(metadata_path).stem + "_gallery.csv")
        gallery_cols = [image_col]
        if image_id_col and image_id_col in image_df.columns:
            gallery_cols.append(image_id_col)
        if text_col in image_df.columns:
            gallery_cols.append(text_col)
        if caption_id_col and caption_id_col in image_df.columns:
            gallery_cols.append(caption_id_col)
        image_df[gallery_cols].to_csv(gallery_path, index=False)
        print(f"Saved metadata to {metadata_path}")
        print(f"Saved gallery metadata to {gallery_path}")
    print(f"Saved image features to {image_feature_path}")
    print(f"Saved text features to {text_feature_path}")


if __name__ == "__main__":
    main()
