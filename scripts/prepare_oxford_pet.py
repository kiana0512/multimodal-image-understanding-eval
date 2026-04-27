from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm
from shutil import copyfile


def trimap_to_binary(mask: Image.Image) -> Image.Image:
    """Convert Oxford Pet trimap to foreground binary mask."""
    arr = np.asarray(mask)
    # Oxford-IIIT Pet trimap: 1 pet, 2 background, 3 border. Treat pet+border as foreground.
    binary = np.where(arr != 2, 255, 0).astype(np.uint8)
    return Image.fromarray(binary)


def make_pseudo_mask(gt_mask: Image.Image, mode: str, rng: np.random.Generator) -> Image.Image:
    """Create simulated pseudo masks for quality-evaluation demos."""
    mask = trimap_to_binary(gt_mask)
    if mode == "erode":
        return mask.filter(ImageFilter.MinFilter(5))
    if mode == "dilate":
        return mask.filter(ImageFilter.MaxFilter(5))
    if mode == "blur_threshold":
        arr = np.asarray(mask.filter(ImageFilter.GaussianBlur(radius=2)), dtype=np.uint8)
        return Image.fromarray(np.where(arr > 128, 255, 0).astype(np.uint8))
    if mode == "shift":
        arr = np.asarray(mask)
        shifted = np.roll(arr, shift=int(rng.integers(-8, 9)), axis=1)
        shifted = np.roll(shifted, shift=int(rng.integers(-8, 9)), axis=0)
        return Image.fromarray(shifted.astype(np.uint8))
    if mode == "holes":
        arr = np.asarray(mask).copy()
        h, w = arr.shape
        for _ in range(5):
            y = int(rng.integers(0, h)); x = int(rng.integers(0, w)); r = int(rng.integers(5, 16))
            yy, xx = np.ogrid[:h, :w]
            arr[(yy - y) ** 2 + (xx - x) ** 2 <= r ** 2] = 0
        return Image.fromarray(arr.astype(np.uint8))
    return mask


def save_image_like(obj, path: Path) -> None:
    """Save a PIL image object to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(obj, "save"):
        obj.save(path)
    else:
        Image.open(obj).save(path)


def main() -> None:
    """Prepare Oxford-IIIT Pet classification and segmentation manifests."""
    parser = argparse.ArgumentParser(description="Prepare Oxford-IIIT Pet manifests and simulated pseudo masks.")
    parser.add_argument("--split", default="trainval", choices=["trainval", "test"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="data/processed/oxford_pet")
    parser.add_argument("--data-root", default="data/downloads")
    parser.add_argument("--download", action="store_true", default=True)
    parser.add_argument("--make-pseudo-masks", action="store_true")
    args = parser.parse_args()

    try:
        from torchvision.datasets import OxfordIIITPet
    except Exception as exc:
        raise ImportError("torchvision is required for Oxford-IIIT Pet. Create the `llm` env first.") from exc

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    mask_dir = output_dir / "masks"
    pseudo_dir = output_dir / "pseudo_masks"
    rng = np.random.default_rng(42)

    cls_ds = OxfordIIITPet(root=args.data_root, split=args.split, target_types="category", download=args.download)
    seg_ds = OxfordIIITPet(root=args.data_root, split=args.split, target_types="segmentation", download=args.download)
    class_names = getattr(cls_ds, "classes", [])

    cls_rows = []
    seg_rows = []
    total = min(len(cls_ds), args.max_samples)
    modes = ["erode", "dilate", "blur_threshold", "shift", "holes"]
    for idx in tqdm(range(total), desc="prepare oxford pet"):
        image, class_id = cls_ds[idx]
        _, trimap = seg_ds[idx]
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        stem = f"{idx:06d}_{class_name.replace(' ', '_')}"
        image_path = image_dir / f"{stem}.jpg"
        imagefolder_path = output_dir / "imagefolder" / class_name / f"{stem}.jpg"
        gt_path = mask_dir / f"{stem}_gt.png"
        pseudo_path = pseudo_dir / f"{stem}_pseudo.png"
        save_image_like(image.convert("RGB"), image_path)
        imagefolder_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(image_path, imagefolder_path)
        gt_binary = trimap_to_binary(trimap)
        save_image_like(gt_binary, gt_path)
        if args.make_pseudo_masks:
            pseudo = make_pseudo_mask(trimap, modes[idx % len(modes)], rng)
        else:
            pseudo = gt_binary
        save_image_like(pseudo, pseudo_path)
        cls_rows.append({
            "image_path": image_path.as_posix(),
            "label": class_name,
            "split": args.split,
            "source": "torchvision:OxfordIIITPet",
            "class_id": class_id,
            "class_name": class_name,
        })
        seg_rows.append({
            "image_path": image_path.as_posix(),
            "gt_mask_path": gt_path.as_posix(),
            "pred_mask_path": pseudo_path.as_posix(),
            "split": args.split,
            "dataset": "oxford_pet_simulated_pseudo_masks",
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    cls_manifest = output_dir / "oxford_pet_classification_manifest.csv"
    seg_manifest = output_dir / "oxford_pet_segmentation_manifest.csv"
    pd.DataFrame(cls_rows).to_csv(cls_manifest, index=False)
    pd.DataFrame(seg_rows).to_csv(seg_manifest, index=False)
    print(f"Saved classification manifest: {cls_manifest}")
    print(f"Saved segmentation manifest: {seg_manifest}")
    print("Pseudo masks are simulated for evaluation-flow validation; they are not SAM3/MedSAM outputs.")


if __name__ == "__main__":
    main()
