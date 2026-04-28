from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import ssl
import urllib.error
from shutil import copyfile
from urllib.error import URLError

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from tqdm import tqdm

NETWORK_ERRORS = (
    ssl.SSLError,
    urllib.error.URLError,
    URLError,
    ConnectionError,
    TimeoutError,
    OSError,
    RuntimeError,
)


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


def rel(path: Path) -> str:
    """Return a stable project-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def output_manifest_paths(output_root: Path) -> tuple[Path, Path]:
    """Return classification and segmentation manifest paths."""
    return (
        output_root / "oxford_pet_classification_manifest.csv",
        output_root / "oxford_pet_segmentation_manifest.csv",
    )


def find_oxford_pet_root(root: Path) -> Path | None:
    """
    Find an extracted Oxford-IIIT Pet directory under root.

    Expected layout:
    - images/
    - annotations/trimaps/
    - annotations/trainval.txt
    - annotations/test.txt
    """
    root = Path(root)
    if not root.exists():
        return None

    candidates = [root]
    candidates.extend(p for p in root.rglob("*") if p.is_dir() and p.name.lower() in {"oxford-iiit-pet", "oxford_pet"})
    for candidate in candidates:
        annotations = candidate / "annotations"
        if (
            (candidate / "images").is_dir()
            and (annotations / "trimaps").is_dir()
            and (annotations / "trainval.txt").is_file()
            and (annotations / "test.txt").is_file()
        ):
            return candidate
    return None


def print_manual_download_help() -> None:
    """Print manual download and fallback instructions."""
    print(
        "\nOxford-IIIT Pet data is not ready.\n"
        "\n"
        "If this is a network or SSL download problem, the model code is not broken.\n"
        "Use one of these options:\n"
        "\n"
        "A. Manual Oxford Pet download, then local build:\n"
        "   1. Download and extract Oxford-IIIT Pet so the folder contains:\n"
        "      data/raw/oxford_pet/images/\n"
        "      data/raw/oxford_pet/annotations/trimaps/\n"
        "      data/raw/oxford_pet/annotations/trainval.txt\n"
        "      data/raw/oxford_pet/annotations/test.txt\n"
        "   2. Run:\n"
        "      python scripts/prepare_oxford_pet.py --source local --local-root data/raw/oxford_pet --max-samples 200 --make-pseudo-masks\n"
        "\n"
        "B. Try torchvision download again:\n"
        "   python scripts/prepare_oxford_pet.py --source torchvision --download --max-samples 200 --make-pseudo-masks\n"
        "\n"
        "More details: README.md\n"
    )


def class_name_from_stem(stem: str) -> str:
    """Infer class name from Oxford Pet image stem."""
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 and parts[1].isdigit() else stem


def read_split_rows(oxford_root: Path, split: str, max_samples: int) -> list[dict[str, object]]:
    """Read Oxford Pet split file rows."""
    split_file = oxford_root / "annotations" / f"{split}.txt"
    rows: list[dict[str, object]] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        stem = parts[0]
        class_id = int(parts[1]) - 1 if len(parts) > 1 and parts[1].isdigit() else -1
        class_name = class_name_from_stem(stem)
        rows.append({"stem": stem, "class_id": class_id, "class_name": class_name})
        if len(rows) >= max_samples:
            break
    return rows


def write_manifests(cls_rows: list[dict[str, object]], seg_rows: list[dict[str, object]], output_root: Path) -> None:
    """Write Oxford Pet classification and segmentation manifests."""
    output_root.mkdir(parents=True, exist_ok=True)
    cls_manifest, seg_manifest = output_manifest_paths(output_root)
    pd.DataFrame(cls_rows).to_csv(cls_manifest, index=False)
    pd.DataFrame(seg_rows).to_csv(seg_manifest, index=False)
    print(f"Saved classification manifest: {rel(cls_manifest)}")
    print(f"Saved segmentation manifest: {rel(seg_manifest)}")
    print("Pseudo masks are simulated for evaluation-flow validation; they are not SAM/MedSAM outputs.")


def build_from_local_extracted(oxford_root: Path, args: argparse.Namespace) -> None:
    """Build manifests from an already extracted Oxford-IIIT Pet directory."""
    output_root = Path(args.output_root)
    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    pseudo_dir = output_root / "pseudo_masks"
    rng = np.random.default_rng(42)
    modes = ["erode", "dilate", "blur_threshold", "shift", "holes"]
    cls_rows: list[dict[str, object]] = []
    seg_rows: list[dict[str, object]] = []

    split_rows = read_split_rows(oxford_root, args.split, args.max_samples)
    if not split_rows:
        raise ValueError(f"No rows found in {oxford_root / 'annotations' / (args.split + '.txt')}")

    for idx, item in enumerate(tqdm(split_rows, desc="prepare oxford pet local")):
        stem = str(item["stem"])
        class_name = str(item["class_name"])
        src_image = oxford_root / "images" / f"{stem}.jpg"
        src_trimap = oxford_root / "annotations" / "trimaps" / f"{stem}.png"
        if not src_image.exists() or not src_trimap.exists():
            raise FileNotFoundError(f"Missing image or trimap for Oxford Pet sample: {stem}")

        safe_class_name = class_name.replace(" ", "_")
        out_stem = f"{idx:06d}_{safe_class_name}"
        image_path = image_dir / f"{out_stem}.jpg"
        imagefolder_path = output_root / "imagefolder" / class_name / f"{out_stem}.jpg"
        gt_path = mask_dir / f"{out_stem}_gt.png"
        pseudo_path = pseudo_dir / f"{out_stem}_pseudo.png"

        image_path.parent.mkdir(parents=True, exist_ok=True)
        imagefolder_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(src_image, image_path)
        copyfile(src_image, imagefolder_path)

        trimap = Image.open(src_trimap)
        gt_binary = trimap_to_binary(trimap)
        save_image_like(gt_binary, gt_path)
        pseudo = make_pseudo_mask(trimap, modes[idx % len(modes)], rng) if args.make_pseudo_masks else gt_binary
        save_image_like(pseudo, pseudo_path)

        cls_rows.append({
            "image_path": rel(image_path),
            "label": class_name,
            "split": args.split,
            "source": f"local:{rel(oxford_root)}",
            "class_id": item["class_id"],
            "class_name": class_name,
        })
        seg_rows.append({
            "image_path": rel(image_path),
            "gt_mask_path": rel(gt_path),
            "pred_mask_path": rel(pseudo_path),
            "split": args.split,
            "dataset": "oxford_pet_simulated_pseudo_masks",
        })

    write_manifests(cls_rows, seg_rows, output_root)


def build_from_torchvision(args: argparse.Namespace) -> None:
    """Build manifests with torchvision.datasets.OxfordIIITPet."""
    try:
        from torchvision.datasets import OxfordIIITPet
    except Exception as exc:
        raise ImportError("torchvision is required for Oxford-IIIT Pet. Create or activate the `llm` env first.") from exc

    output_root = Path(args.output_root)
    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    pseudo_dir = output_root / "pseudo_masks"
    rng = np.random.default_rng(42)
    modes = ["erode", "dilate", "blur_threshold", "shift", "holes"]

    cls_ds = OxfordIIITPet(
        root=args.data_root,
        split=args.split,
        target_types="category",
        download=args.download,
    )
    seg_ds = OxfordIIITPet(
        root=args.data_root,
        split=args.split,
        target_types="segmentation",
        download=args.download,
    )
    class_names = getattr(cls_ds, "classes", [])
    cls_rows: list[dict[str, object]] = []
    seg_rows: list[dict[str, object]] = []
    total = min(len(cls_ds), args.max_samples)

    for idx in tqdm(range(total), desc="prepare oxford pet torchvision"):
        image, class_id = cls_ds[idx]
        _, trimap = seg_ds[idx]
        class_name = class_names[class_id] if class_names and class_id < len(class_names) else str(class_id)
        safe_class_name = class_name.replace(" ", "_")
        stem = f"{idx:06d}_{safe_class_name}"
        image_path = image_dir / f"{stem}.jpg"
        imagefolder_path = output_root / "imagefolder" / class_name / f"{stem}.jpg"
        gt_path = mask_dir / f"{stem}_gt.png"
        pseudo_path = pseudo_dir / f"{stem}_pseudo.png"

        save_image_like(image.convert("RGB"), image_path)
        imagefolder_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(image_path, imagefolder_path)
        gt_binary = trimap_to_binary(trimap)
        save_image_like(gt_binary, gt_path)
        pseudo = make_pseudo_mask(trimap, modes[idx % len(modes)], rng) if args.make_pseudo_masks else gt_binary
        save_image_like(pseudo, pseudo_path)

        cls_rows.append({
            "image_path": rel(image_path),
            "label": class_name,
            "split": args.split,
            "source": "torchvision:OxfordIIITPet",
            "class_id": class_id,
            "class_name": class_name,
        })
        seg_rows.append({
            "image_path": rel(image_path),
            "gt_mask_path": rel(gt_path),
            "pred_mask_path": rel(pseudo_path),
            "split": args.split,
            "dataset": "oxford_pet_simulated_pseudo_masks",
        })

    write_manifests(cls_rows, seg_rows, output_root)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare Oxford-IIIT Pet manifests with online, local, or auto source detection."
    )
    parser.add_argument("--source", default="auto", choices=["auto", "torchvision", "local"])
    parser.add_argument("--split", default="trainval", choices=["trainval", "test"])
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--data-root", default="data/raw/oxford_pet")
    parser.add_argument("--local-root", default=None)
    parser.add_argument("--output-root", "--output-dir", dest="output_root", default="data/processed/oxford_pet")
    parser.add_argument("--download", dest="download", action="store_true", default=True)
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument("--make-pseudo-masks", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate processed files and manifests.")
    return parser.parse_args()


def main() -> None:
    """Prepare Oxford-IIIT Pet classification and segmentation manifests."""
    args = parse_args()
    output_root = Path(args.output_root)
    _, seg_manifest = output_manifest_paths(output_root)
    if seg_manifest.exists() and not args.force:
        print(f"Oxford Pet processed manifest already exists: {rel(seg_manifest)}")
        print("Use --force to regenerate it.")
        return

    local_search_roots = [Path(args.local_root or args.data_root)]
    if args.local_root and Path(args.local_root) != Path(args.data_root):
        local_search_roots.append(Path(args.data_root))

    if args.source in {"auto", "local"}:
        for search_root in local_search_roots:
            oxford_root = find_oxford_pet_root(search_root)
            if oxford_root is not None:
                print(f"Found local Oxford Pet data: {rel(oxford_root)}")
                build_from_local_extracted(oxford_root, args)
                return
        if args.source == "local":
            print(f"Could not find extracted Oxford Pet data under: {args.local_root or args.data_root}")
            print_manual_download_help()
            raise SystemExit(2)

    if args.source == "auto" and not args.download:
        print("Local Oxford Pet data was not found and --no-download was set.")
        print_manual_download_help()
        raise SystemExit(2)

    try:
        print(f"Using torchvision OxfordIIITPet with download={args.download}.")
        build_from_torchvision(args)
    except NETWORK_ERRORS as exc:
        print("\nOxford Pet download/load failed before data preparation could finish.")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error message: {exc}")
        print_manual_download_help()
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
