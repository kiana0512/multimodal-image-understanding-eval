"""Heuristic AIGC image quality metrics.

These scores are proxy metrics for screening and badcase mining, not a replacement for
human review or a trained aesthetic model.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from mm_eval.retrieval.similarity import pairwise_cosine_similarity

def _read_gray_image(image_path: str | Path) -> np.ndarray:
    """Read an image as grayscale using OpenCV when available, otherwise PIL."""
    try:
        import cv2
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            return image.astype(np.float32)
    except ImportError:
        pass
    try:
        return np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    except Exception as exc:
        raise FileNotFoundError(f"Image not found: {image_path}") from exc

def blur_score(image_path: str | Path) -> float:
    """Return normalized Laplacian variance blur score."""
    image=_read_gray_image(image_path)
    try:
        import cv2
        variance = float(cv2.Laplacian(image.astype(np.uint8), cv2.CV_64F).var())
    except ImportError:
        # NumPy fallback: gradient variance is less standard than Laplacian variance,
        # but it keeps the toy pipeline runnable without OpenCV.
        gy, gx = np.gradient(image)
        variance = float((gx * gx + gy * gy).var())
    return float(min(variance/1000.0,1.0))

def brightness_contrast_scores(image_path: str | Path) -> tuple[float,float]:
    """Return brightness and contrast scores in [0, 1]."""
    image=_read_gray_image(image_path)
    mean=float(image.mean())/255.0; std=float(image.std())/128.0
    return max(0.0,1.0-abs(mean-0.5)*2.0), float(min(std,1.0))

def resolution_check(image_path: str | Path, min_size: int = 256) -> float:
    """Return 1 if both sides meet min_size, otherwise scaled score."""
    with Image.open(image_path) as img: w,h=img.size
    return float(min(min(w,h)/min_size,1.0))

def aspect_ratio_check(image_path: str | Path, min_ratio: float = 0.5, max_ratio: float = 2.0) -> float:
    """Return 1 if aspect ratio is within range."""
    with Image.open(image_path) as img: w,h=img.size
    ratio=w/max(h,1); return 1.0 if min_ratio <= ratio <= max_ratio else 0.0

def duplicate_scores(features: np.ndarray | None, threshold: float = 0.98) -> np.ndarray:
    """Return duplicate risk score per item using feature similarity."""
    if features is None or len(features)==0: return np.zeros(0,dtype=np.float32)
    sims=pairwise_cosine_similarity(features,features); np.fill_diagonal(sims,-1.0)
    return (sims.max(axis=1) >= threshold).astype(np.float32)

def diversity_score(features: np.ndarray | None) -> float:
    """Return simple diversity score based on pairwise dissimilarity."""
    if features is None or len(features)<=1: return 0.0
    sims=pairwise_cosine_similarity(features,features); upper=sims[np.triu_indices_from(sims,k=1)]
    return float(np.clip(1.0-upper.mean(),0.0,1.0))

def make_badcase_reason(row: pd.Series) -> str:
    """Generate readable badcase tags."""
    reasons=[]
    if row.get("clip_score",1.0)<0.25: reasons.append("low_clip_score")
    if row.get("blur_score",1.0)<0.2: reasons.append("blurry")
    if row.get("brightness_score",1.0)<0.3: reasons.append("bad_brightness")
    if row.get("contrast_score",1.0)<0.2: reasons.append("low_contrast")
    if row.get("duplicate_score",0.0)>0.5: reasons.append("duplicate")
    if row.get("resolution_score",1.0)<1.0: reasons.append("low_resolution")
    if row.get("aspect_ratio_score",1.0)<1.0: reasons.append("bad_aspect_ratio")
    return ";".join(reasons) if reasons else "ok"

def compute_overall_score(row: pd.Series, weights: dict[str,float] | None = None) -> float:
    """Compute heuristic weighted quality score."""
    weights=weights or {"clip_score":0.35,"blur_score":0.2,"brightness_score":0.15,"contrast_score":0.15,"resolution_score":0.1,"aspect_ratio_score":0.05}
    total=sum(weights.values()); score=sum(float(row.get(k,0.0))*v for k,v in weights.items())/max(total,1e-8)
    return float(np.clip(score - 0.2*float(row.get("duplicate_score",0.0)),0.0,1.0))
