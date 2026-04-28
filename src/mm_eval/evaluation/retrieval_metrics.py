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
    out={f"recall_at_{k}": recall_at_k(scores,target_indices,k) for k in ks}
    out.update({"mrr": mean_reciprocal_rank(scores,target_indices), "mean_rank": mean_rank(scores,target_indices), "top1_accuracy": top1_accuracy(scores,target_indices)})
    return out

def ranks_from_positive_matrix(scores: np.ndarray, positive: np.ndarray) -> np.ndarray:
    """Return first positive rank per query from a boolean positive matrix."""
    scores = np.asarray(scores)
    positive = np.asarray(positive, dtype=bool)
    order = np.argsort(-scores, axis=1)
    ranks = []
    for i in range(scores.shape[0]):
        ranked_positive = positive[i, order[i]]
        positions = np.where(ranked_positive)[0]
        ranks.append(int(positions[0]) + 1 if len(positions) else scores.shape[1] + 1)
    return np.asarray(ranks)

def summarize_retrieval_by_positives(
    scores: np.ndarray,
    positive: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Summarize retrieval when each query can have one or more positive gallery items."""
    ranks = ranks_from_positive_matrix(scores, positive)
    out = {f"recall_at_{k}": float(np.mean(ranks <= k)) for k in ks}
    out.update({
        "mrr": float(np.mean(1.0 / ranks)),
        "mean_rank": float(np.mean(ranks)),
        "top1_accuracy": float(np.mean(ranks <= 1)),
        "num_queries": float(scores.shape[0]),
        "num_gallery": float(scores.shape[1]),
    })
    return out
