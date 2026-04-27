"""Classification evaluation loop."""
from __future__ import annotations
import torch

def evaluate_classifier(model, dataloader, device: str = "cpu") -> tuple[list[int], list[int]]:
    """Return labels and predictions for a classifier."""
    model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for images,labels in dataloader:
            images=images.to(device); logits=model(images); preds=logits.argmax(dim=1).cpu().tolist()
            y_pred.extend(preds); y_true.extend(labels.tolist())
    return y_true,y_pred
