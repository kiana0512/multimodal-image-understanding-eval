"""Retrieval metric implementations."""
from __future__ import annotations
import numpy as np

def _ranks_from_scores(scores: np.ndarray, target_indices: np.ndarray) -> np.ndarray:
    order=np.argsort(-scores,axis=1); ranks=[]
    for i,target in enumerate(target_indices):
        pos=np.where(order[i] == target)[0]
        ranks.append(int(pos[0])+1 if len(pos) else scores.shape[1]+1)
    return np.asarray(ranks)

def recall_at_k(scores: np.ndarray, target_indices: list[int] | np.ndarray, k: int = 1) -> float:
    """Compute Recall@K for one positive target per query."""
    return float(np.mean(_ranks_from_scores(np.asarray(scores), np.asarray(target_indices)) <= k))

def mean_reciprocal_rank(scores: np.ndarray, target_indices: list[int] | np.ndarray) -> float:
    """Compute MRR."""
    ranks=_ranks_from_scores(np.asarray(scores),np.asarray(target_indices)); return float(np.mean(1.0/ranks))

def mean_rank(scores: np.ndarray, target_indices: list[int] | np.ndarray) -> float:
    """Compute mean rank."""
    return float(np.mean(_ranks_from_scores(np.asarray(scores), np.asarray(target_indices))))

def top1_accuracy(scores: np.ndarray, target_indices: list[int] | np.ndarray) -> float:
    """Compute Top-1 retrieval accuracy."""
    return recall_at_k(scores,target_indices,k=1)

def summarize_retrieval(scores: np.ndarray, target_indices: list[int] | np.ndarray, ks: tuple[int,...]=(1,5,10)) -> dict[str,float]:
    """Return common retrieval metrics."""
    out={f"recall@{k}": recall_at_k(scores,target_indices,k) for k in ks}
    out.update({"mrr": mean_reciprocal_rank(scores,target_indices), "mean_rank": mean_rank(scores,target_indices), "top1_accuracy": top1_accuracy(scores,target_indices)})
    return out
