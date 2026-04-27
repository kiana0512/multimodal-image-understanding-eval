"""Lightweight image classification baseline."""
from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from mm_eval.models.image_backbone import build_image_backbone
from mm_eval.utils.device import get_device


def build_demo_dataset(dataset_name: str, data_root: str, image_size: int, train: bool = True, download: bool = False):
    """Build CIFAR10 or ImageFolder dataset."""
    transform=transforms.Compose([transforms.Resize((image_size,image_size)), transforms.ToTensor()])
    if dataset_name.lower()=="cifar10": return datasets.CIFAR10(root=data_root,train=train,download=download,transform=transform)
    if dataset_name.lower()=="imagefolder": return datasets.ImageFolder(root=data_root,transform=transform)
    raise ValueError(f"Unsupported dataset_name: {dataset_name}")


def run_classification_demo(config: dict) -> dict[str,float]:
    """Run a small CPU-friendly classification demo."""
    device=get_device(config.get("device")); data_root=config.get("data_root","data/downloads"); image_size=int(config.get("image_size",64))
    dataset=build_demo_dataset(config.get("dataset_name","CIFAR10"),data_root,image_size,train=True,download=bool(config.get("download",False)))
    max_samples=int(config.get("max_samples",256)); indices=list(range(min(len(dataset),max_samples))); subset=Subset(dataset,indices)
    loader=DataLoader(subset,batch_size=int(config.get("batch_size",16)),shuffle=True)
    model=build_image_backbone(config.get("model_name","resnet18"),num_classes=int(config.get("num_classes",10)),pretrained=bool(config.get("use_pretrained",False))).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=float(config.get("lr",1e-3))); loss_fn=torch.nn.CrossEntropyLoss(); epochs=int(config.get("epochs",1))
    last_loss=0.0
    for _ in range(epochs):
        model.train()
        for images,labels in loader:
            images,labels=images.to(device),labels.to(device); opt.zero_grad(); loss=loss_fn(model(images),labels); loss.backward(); opt.step(); last_loss=float(loss.item())
    out=Path(config.get("output_dir","outputs/classification")); out.mkdir(parents=True,exist_ok=True); torch.save(model.state_dict(), out/"classification_demo.pth")
    return {"final_loss":last_loss,"num_samples":len(subset)}
