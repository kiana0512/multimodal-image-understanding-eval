"""Image transform factory functions."""
from __future__ import annotations

def build_image_transform(image_size: int = 224, train: bool = False):
    """Build a torchvision transform for RGB images."""
    from torchvision import transforms
    ops=[]
    if train: ops += [transforms.RandomResizedCrop(image_size), transforms.RandomHorizontalFlip()]
    else: ops += [transforms.Resize((image_size, image_size))]
    ops += [transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
    return transforms.Compose(ops)
