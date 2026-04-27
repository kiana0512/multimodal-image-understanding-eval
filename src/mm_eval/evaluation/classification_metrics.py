"""Classification metrics."""
from __future__ import annotations
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def classification_summary(y_true: list[int], y_pred: list[int], labels: list[str] | None = None) -> dict[str, object]:
    """Compute accuracy, confusion matrix and text report."""
    return {"accuracy": float(accuracy_score(y_true,y_pred)), "confusion_matrix": confusion_matrix(y_true,y_pred).tolist(), "report": classification_report(y_true,y_pred,target_names=labels,zero_division=0)}
