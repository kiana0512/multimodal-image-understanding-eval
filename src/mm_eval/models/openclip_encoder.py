"""OpenCLIP encoder wrapper."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
from mm_eval.utils.device import get_device
from mm_eval.retrieval.similarity import l2_normalize

class OpenClipEncoder:
    """Batch image/text encoder backed by open_clip_torch."""
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str | None = None):
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError("open_clip_torch is not installed. Install it with `pip install open_clip_torch` to use OpenClipEncoder.") from exc
        self.open_clip=open_clip; self.torch=torch; self.device=get_device(device)
        self.model,_,self.preprocess=open_clip.create_model_and_transforms(model_name,pretrained=pretrained,device=self.device)
        self.tokenizer=open_clip.get_tokenizer(model_name); self.model.eval()

    def encode_images(self, image_paths: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode image paths into L2-normalized features."""
        feats=[]
        with self.torch.no_grad():
            for i in tqdm(range(0,len(image_paths),batch_size),desc="encode images"):
                batch=[self.preprocess(Image.open(p).convert("RGB")) for p in image_paths[i:i+batch_size]]
                tensor=self.torch.stack(batch).to(self.device)
                feats.append(self.model.encode_image(tensor).detach().cpu().numpy())
        return l2_normalize(np.concatenate(feats,axis=0)) if feats else np.empty((0,0),dtype=np.float32)

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into L2-normalized features."""
        feats=[]
        with self.torch.no_grad():
            for i in tqdm(range(0,len(texts),batch_size),desc="encode texts"):
                tokens=self.tokenizer(texts[i:i+batch_size]).to(self.device)
                feats.append(self.model.encode_text(tokens).detach().cpu().numpy())
        return l2_normalize(np.concatenate(feats,axis=0)) if feats else np.empty((0,0),dtype=np.float32)

    @staticmethod
    def save_features(features: np.ndarray, path: str | Path) -> None:
        """Save features as .npy."""
        p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); np.save(p,features)
