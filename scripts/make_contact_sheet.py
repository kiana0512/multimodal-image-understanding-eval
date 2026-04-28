from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import pandas as pd
from mm_eval.visualization.contact_sheet import make_contact_sheet

def main() -> None:
    """Create a contact sheet from CSV image paths."""
    parser=argparse.ArgumentParser(description="Make contact sheet from a CSV column or image list.")
    parser.add_argument("--csv"); parser.add_argument("--image-column",default="image_path"); parser.add_argument("--caption-column"); parser.add_argument("--max-images",type=int,default=24); parser.add_argument("--images",nargs="*"); parser.add_argument("--output",default="outputs/figures/contact_sheet_demo.png")
    args=parser.parse_args()
    paths=args.images or []; captions=None
    if args.csv:
        df=pd.read_csv(args.csv).dropna(subset=[args.image_column])
        df=df.drop_duplicates(subset=[args.image_column]).head(args.max_images)
        paths=df[args.image_column].astype(str).tolist()
        if args.caption_column and args.caption_column in df.columns:
            captions=df[args.caption_column].fillna("").astype(str).tolist()
    make_contact_sheet(paths,args.output,captions=captions); print(args.output)
if __name__ == "__main__": main()
