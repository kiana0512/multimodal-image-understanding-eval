from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from mm_eval.data.manifest import build_image_folder_manifest, load_manifest, validate_manifest, check_image_paths, save_manifest

def main() -> None:
    """Build or validate a manifest."""
    parser=argparse.ArgumentParser(description="Build or validate image manifest CSV.")
    parser.add_argument("--image-root", type=str, help="Image folder root to scan.")
    parser.add_argument("--input", type=str, help="Existing manifest to validate.")
    parser.add_argument("--output", type=str, required=True, help="Output clean manifest CSV.")
    parser.add_argument("--drop-missing", action="store_true", help="Drop rows whose image files are missing.")
    args=parser.parse_args()
    if args.image_root:
        df=build_image_folder_manifest(args.image_root,args.output)
    elif args.input:
        df=validate_manifest(load_manifest(args.input),["image_path"])
        df=check_image_paths(df,drop_missing=args.drop_missing)
        save_manifest(df,args.output)
    else:
        parser.error("Provide --image-root or --input.")
    print(f"Saved {len(df)} rows to {args.output}")
if __name__ == "__main__": main()
