"""Segmentation and pseudo-mask quality metrics."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

def _require_cv2():
    """Import OpenCV only when image-file or boundary operations need it."""
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required for mask file IO and boundary F1. Install it with `pip install opencv-python`.") from exc

def binarize_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Convert a 0/1 or 0/255 mask to boolean."""
    arr=mask.astype(np.float32)
    if arr.max(initial=0)>1.0: arr=arr/255.0
    return arr >= threshold

def read_mask(path: str | Path, threshold: float = 0.5) -> np.ndarray:
    """Read a mask image and return boolean mask."""
    cv2 = _require_cv2()
    mask=cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None: raise FileNotFoundError(f"Failed to read mask: {path}")
    return binarize_mask(mask,threshold)

def dice_score(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-7) -> float:
    """Compute Dice score."""
    g,p=binarize_mask(gt),binarize_mask(pred); inter=np.logical_and(g,p).sum()
    return float((2*inter+eps)/(g.sum()+p.sum()+eps))

def iou_score(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-7) -> float:
    """Compute IoU."""
    g,p=binarize_mask(gt),binarize_mask(pred); inter=np.logical_and(g,p).sum(); union=np.logical_or(g,p).sum()
    return float((inter+eps)/(union+eps))

def precision_recall_f1(gt: np.ndarray, pred: np.ndarray, eps: float = 1e-7) -> tuple[float,float,float]:
    """Compute pixel precision, recall and F1."""
    g,p=binarize_mask(gt),binarize_mask(pred); tp=np.logical_and(g,p).sum(); fp=np.logical_and(~g,p).sum(); fn=np.logical_and(g,~p).sum()
    precision=float((tp+eps)/(tp+fp+eps)); recall=float((tp+eps)/(tp+fn+eps)); f1=float((2*precision*recall+eps)/(precision+recall+eps))
    return precision,recall,f1

def boundary_f1_score(gt: np.ndarray, pred: np.ndarray, dilation: int = 1) -> float:
    """Compute simplified boundary F1 using morphology."""
    cv2 = _require_cv2()
    g,p=binarize_mask(gt).astype(np.uint8),binarize_mask(pred).astype(np.uint8); kernel=np.ones((3,3),np.uint8)
    gb=cv2.morphologyEx(g,cv2.MORPH_GRADIENT,kernel); pb=cv2.morphologyEx(p,cv2.MORPH_GRADIENT,kernel)
    if dilation>0: gb=cv2.dilate(gb,kernel,iterations=dilation); pb=cv2.dilate(pb,kernel,iterations=dilation)
    return precision_recall_f1(gb,pb)[2]

def evaluate_mask_pair(gt_mask_path: str | Path, pred_mask_path: str | Path, threshold: float = 0.5) -> dict[str,float]:
    """Evaluate one GT/pred mask pair."""
    gt=read_mask(gt_mask_path,threshold); pred=read_mask(pred_mask_path,threshold)
    if gt.shape != pred.shape: raise ValueError(f"Mask shape mismatch: {gt.shape} vs {pred.shape}")
    precision,recall,f1=precision_recall_f1(gt,pred)
    return {"dice":dice_score(gt,pred),"iou":iou_score(gt,pred),"precision":precision,"recall":recall,"f1":f1,"boundary_f1_simple":boundary_f1_score(gt,pred)}

def evaluate_segmentation_manifest(manifest: str | Path, output_csv: str | Path, gt_col: str = "gt_mask_path", pred_col: str = "pred_mask_path", threshold: float = 0.5) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Evaluate mask pairs from manifest and save CSV outputs."""
    df=pd.read_csv(manifest); rows=[]
    for _,row in df.iterrows(): rows.append({**row.to_dict(), **evaluate_mask_pair(row[gt_col],row[pred_col],threshold)})
    result=pd.DataFrame(rows); out=Path(output_csv); out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False)
    cols=["dice","iou","precision","recall","f1","boundary_f1_simple"]; summary=result[cols].agg(["mean","std","min","max"]).reset_index().rename(columns={"index":"stat"})
    summary.to_csv(out.with_name(out.stem+"_summary.csv"),index=False); return result,summary
