"""Manifest loading, validation and construction utilities."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a CSV manifest."""
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Manifest not found: {p}")
    df=pd.read_csv(p)
    if df.empty: raise ValueError(f"Manifest is empty: {p}")
    return df

def validate_manifest(df: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Validate columns and remove rows with missing required values."""
    missing=[c for c in required_columns if c not in df.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    out=df.dropna(subset=required_columns).copy()
    if out.empty: raise ValueError("No valid rows remain after filtering missing required fields.")
    return out.reset_index(drop=True)

def check_image_paths(df: pd.DataFrame, image_column: str = "image_path", root: str | Path = ".", drop_missing: bool = False) -> pd.DataFrame:
    """Check image paths and optionally drop missing images."""
    if image_column not in df.columns: raise ValueError(f"Missing image column: {image_column}")
    base=Path(root); out=df.copy(); out["image_exists"]=[(base/str(p)).exists() for p in out[image_column]]
    missing=int((~out["image_exists"]).sum())
    if missing and not drop_missing: raise FileNotFoundError(f"{missing} image files are missing. Set drop_missing=True to filter them.")
    return out[out["image_exists"]].reset_index(drop=True) if drop_missing else out

def save_manifest(df: pd.DataFrame, path: str | Path) -> None:
    """Save a manifest CSV."""
    p=Path(path); p.parent.mkdir(parents=True, exist_ok=True); df.to_csv(p, index=False)

def filter_split(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Return only selected split when provided."""
    if split is None: return df
    if "split" not in df.columns: raise ValueError("Manifest has no split column.")
    return df[df["split"] == split].reset_index(drop=True)

def build_image_folder_manifest(root: str | Path, output_csv: str | Path, split: str = "val") -> pd.DataFrame:
    """Build an ImageFolder-style manifest."""
    image_root=Path(root)
    if not image_root.exists(): raise FileNotFoundError(f"Image root not found: {image_root}")
    rows=[]
    for p in sorted(image_root.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTENSIONS:
            label=p.parent.name if p.parent != image_root else "unknown"
            rows.append({"image_path": p.as_posix(), "text": label, "label": label, "split": split, "source": "image_folder"})
    if not rows: raise ValueError(f"No images found under {image_root}. Supported extensions: {sorted(IMAGE_EXTENSIONS)}")
    df=pd.DataFrame(rows); save_manifest(df, output_csv); return df

def build_toy_manifest(output_csv: str | Path = "data/sample_manifest.csv") -> pd.DataFrame:
    """Create a tiny manifest with expected toy image paths."""
    rows=[
        {"image_path":"data/toy_images/cat_001.jpg","text":"a cute cat sitting on the sofa","label":"cat","split":"train","source":"toy"},
        {"image_path":"data/toy_images/dog_001.jpg","text":"a dog running on the grass","label":"dog","split":"val","source":"toy"},
        {"image_path":"data/toy_images/sword_icon_001.png","text":"fantasy sword icon, clean ui","label":"ui_icon","split":"val","source":"toy"},
        {"image_path":"data/toy_images/character_001.png","text":"anime game character concept art","label":"character","split":"val","source":"toy"},
    ]
    df=pd.DataFrame(rows); save_manifest(df, output_csv); return df
