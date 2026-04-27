"""Lightweight torchvision image backbone factory."""
from __future__ import annotations

def build_image_backbone(model_name: str = "resnet18", num_classes: int = 10, pretrained: bool = False):
    """Build a torchvision classifier backbone."""
    import torch.nn as nn
    from torchvision import models
    if model_name == "resnet18":
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        model=models.resnet18(weights=weights); model.fc=nn.Linear(model.fc.in_features,num_classes); return model
    if model_name == "mobilenet_v3_small":
        weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model=models.mobilenet_v3_small(weights=weights); model.classifier[-1]=nn.Linear(model.classifier[-1].in_features,num_classes); return model
    raise ValueError(f"Unsupported model_name: {model_name}")
