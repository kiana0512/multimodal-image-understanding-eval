from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path
from PIL import Image, ImageDraw
from mm_eval.data.manifest import build_toy_manifest

def main() -> None:
    """Create tiny placeholder images for pipeline testing."""
    parser=argparse.ArgumentParser(description="Create toy placeholder images. These are not real experiment data.")
    parser.add_argument("--output-dir", default="data/toy_images")
    args=parser.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    specs=[("cat_001.jpg","cat","#e8b4b8"),("dog_001.jpg","dog","#9ec1cf"),("sword_icon_001.png","sword icon","#c9cba3"),("character_001.png","character","#b5ead7")]
    for name,text,color in specs:
        img=Image.new("RGB",(256,256),color); draw=ImageDraw.Draw(img); draw.text((30,115),text,fill="black"); img.save(out/name)
    build_toy_manifest("data/sample_manifest.csv")
    print(f"Created {len(specs)} placeholder images under {out}. Status: demo data only.")
if __name__ == "__main__": main()
