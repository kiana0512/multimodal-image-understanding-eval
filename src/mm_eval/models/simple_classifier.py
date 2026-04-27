"""Simple classifier heads."""
from __future__ import annotations
import torch.nn as nn

class LinearClassifier(nn.Module):
    """A linear classifier for frozen features."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__(); self.fc=nn.Linear(input_dim,num_classes)
    def forward(self, x):
        """Return logits."""
        return self.fc(x)
