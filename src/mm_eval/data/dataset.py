"""PyTorch Dataset implementations."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def _open_rgb(path: Path) -> Image.Image:
    try: return Image.open(path).convert("RGB")
    except Exception as exc: raise RuntimeError(f"Failed to read image: {path}") from exc

class ImageTextDataset(Dataset):
    """Dataset that returns image tensor, text and metadata."""
    def __init__(self, manifest: str | Path | pd.DataFrame, image_column: str = "image_path", text_column: str = "text", transform: Any | None = None):
        self.df=pd.read_csv(manifest) if not isinstance(manifest,pd.DataFrame) else manifest.reset_index(drop=True)
        self.image_column=image_column; self.text_column=text_column; self.transform=transform or transforms.ToTensor()
        for col in [image_column,text_column]:
            if col not in self.df.columns: raise ValueError(f"Missing required column: {col}")
    def __len__(self)->int: return len(self.df)
    def __getitem__(self, idx:int)->dict[str,Any]:
        row=self.df.iloc[idx].to_dict(); path=Path(str(row[self.image_column]))
        return {"image": self.transform(_open_rgb(path)), "text": str(row[self.text_column]), "metadata": row}

class AIGCImageDataset(ImageTextDataset):
    """Dataset for prompt-generated images."""
    def __init__(self, manifest: str | Path | pd.DataFrame, image_column: str = "image_path", prompt_column: str = "prompt", transform: Any | None = None):
        super().__init__(manifest, image_column=image_column, text_column=prompt_column, transform=transform)

class SegmentationPairDataset(Dataset):
    """Dataset that returns mask pair metadata."""
    def __init__(self, manifest: str | Path | pd.DataFrame, image_column: str = "image_path", gt_mask_column: str = "gt_mask_path", pred_mask_column: str = "pred_mask_path"):
        self.df=pd.read_csv(manifest) if not isinstance(manifest,pd.DataFrame) else manifest.reset_index(drop=True)
        for col in [image_column,gt_mask_column,pred_mask_column]:
            if col not in self.df.columns: raise ValueError(f"Missing required column: {col}")
    def __len__(self)->int: return len(self.df)
    def __getitem__(self, idx:int)->dict[str,Any]: return self.df.iloc[idx].to_dict()

class ImageClassificationDataset(ImageTextDataset):
    """Manifest-backed image classification dataset."""
    def __init__(self, manifest: str | Path | pd.DataFrame, image_column: str = "image_path", label_column: str = "label", transform: Any | None = None):
        super().__init__(manifest, image_column=image_column, text_column=label_column, transform=transform)
        self.label_column=label_column; labels=sorted(self.df[label_column].astype(str).unique()); self.class_to_idx={v:i for i,v in enumerate(labels)}
    def __getitem__(self, idx:int)->dict[str,Any]:
        item=super().__getitem__(idx); label=str(item["metadata"][self.label_column]); item["label"]=self.class_to_idx[label]; return item
