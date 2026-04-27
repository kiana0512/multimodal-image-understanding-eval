"""Contact sheet generation."""
from __future__ import annotations
from pathlib import Path
from PIL import Image, ImageDraw

def make_contact_sheet(image_paths: list[str], output_path: str | Path, thumb_size: int = 160, cols: int = 4, captions: list[str] | None = None) -> Path:
    """Create a grid contact sheet for images."""
    if not image_paths: raise ValueError("No image paths provided.")
    rows=(len(image_paths)+cols-1)//cols; sheet=Image.new("RGB",(cols*thumb_size,rows*(thumb_size+24)),"white"); draw=ImageDraw.Draw(sheet)
    for i,path in enumerate(image_paths):
        img=Image.open(path).convert("RGB"); img.thumbnail((thumb_size,thumb_size)); x=(i%cols)*thumb_size; y=(i//cols)*(thumb_size+24)
        sheet.paste(img,(x,y)); draw.text((x+4,y+thumb_size+4),(captions[i] if captions else Path(path).name)[:24],fill="black")
    out=Path(output_path); out.parent.mkdir(parents=True,exist_ok=True); sheet.save(out); return out
