from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

URLS = {
    "train2014": "http://images.cocodataset.org/zips/train2014.zip",
    "val2014": "http://images.cocodataset.org/zips/val2014.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
}

ZIP_NAMES = {
    "train2014": "train2014.zip",
    "val2014": "val2014.zip",
    "annotations": "annotations_trainval2014.zip",
}

MIN_BYTES = {
    "train2014": 10 * 1024**3,
    "val2014": 5 * 1024**3,
    "annotations": 100 * 1024**2,
}

SPACE_HINTS = {
    "train": "train2014 约 13GB，annotations 约 241MB；建议预留 25GB 以上。",
    "val": "val2014 约 6GB，annotations 约 241MB；建议预留 12GB 以上。",
    "trainval": "train2014+val2014+annotations 约 19GB+；完整解压后建议预留 30GB 以上。",
}


def rel(path: Path) -> str:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def required_parts(split: str) -> list[str]:
    """Return required COCO2014 zip parts for a split."""
    if split == "train":
        return ["train2014", "annotations"]
    if split == "val":
        return ["val2014", "annotations"]
    return ["train2014", "val2014", "annotations"]


def expected_outputs(root: Path, part: str) -> list[Path]:
    """Return paths that prove one COCO part is extracted."""
    if part == "annotations":
        return [
            root / "annotations" / "captions_train2014.json",
            root / "annotations" / "captions_val2014.json",
        ]
    return [root / part]


def print_space_hint(root: Path, split: str) -> None:
    """Print disk space guidance and best-effort free space."""
    root.mkdir(parents=True, exist_ok=True)
    print(f"Disk space hint for COCO2014 {split}: {SPACE_HINTS[split]}")
    try:
        usage = shutil.disk_usage(root)
        print(f"Free disk space under {rel(root)}: {usage.free / 1024**3:.1f} GB")
    except Exception:
        print("Could not determine free disk space. Please check manually before downloading.")


def has_aria2c() -> bool:
    """Return whether aria2c is available on PATH."""
    return shutil.which("aria2c") is not None


def download_with_aria2c(url: str, output: Path, workers: int, force: bool) -> None:
    """Download with aria2c using resume and multiple connections."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aria2c",
        "--continue=true",
        f"--max-connection-per-server={workers}",
        f"--split={workers}",
        "--min-split-size=16M",
        "--file-allocation=none",
        "--allow-overwrite=true" if force else "--allow-overwrite=false",
        "--dir",
        str(output.parent),
        "--out",
        output.name,
        url,
    ]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def remote_size(url: str) -> int | None:
    """Return remote content length if the server exposes it."""
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=30) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def download_with_urllib_resume(url: str, output: Path, force: bool) -> None:
    """Download with urllib and HTTP Range resume support."""
    output.parent.mkdir(parents=True, exist_ok=True)
    if force and output.exists():
        output.unlink()

    existing = output.stat().st_size if output.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    mode = "ab" if existing else "wb"
    total = remote_size(url)
    if total and existing >= total:
        print(f"Already downloaded: {rel(output)}")
        return

    print(f"Python fallback download with resume: {url}")
    if existing:
        print(f"Resuming from {existing / 1024**2:.1f} MB")
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        if existing and getattr(response, "status", None) == 200:
            print("Server did not honor resume request; restarting this file.")
            existing = 0
            mode = "wb"
        with output.open(mode) as fh:
            downloaded = existing
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\rDownloaded {downloaded / 1024**3:.2f}/{total / 1024**3:.2f} GB", end="")
                else:
                    print(f"\rDownloaded {downloaded / 1024**3:.2f} GB", end="")
    print()


def validate_zip(path: Path, part: str) -> bool:
    """Check zip existence and rough size."""
    if not path.exists():
        print(f"MISSING: {rel(path)}")
        return False
    size = path.stat().st_size
    min_size = MIN_BYTES[part]
    ok = size >= min_size
    status = "FOUND" if ok else "TOO SMALL"
    print(f"{status}: {rel(path)} ({size / 1024**3:.2f} GB)")
    return ok


def download_part(root: Path, part: str, workers: int, force: bool) -> None:
    """Download one COCO zip part."""
    output = root / ZIP_NAMES[part]
    if output.exists() and not force and validate_zip(output, part):
        print(f"Skipping existing download: {rel(output)}")
        return
    print(f"Downloading {output.name} ...")
    if has_aria2c():
        download_with_aria2c(URLS[part], output, workers, force)
    else:
        print("aria2c not found; falling back to Python downloader.")
        print("For faster downloads on Windows, run: conda install -c conda-forge aria2 -y")
        download_with_urllib_resume(URLS[part], output, force)
    if not validate_zip(output, part):
        raise RuntimeError(f"Downloaded file is missing or too small: {output}")


def extract_part(root: Path, part: str, force: bool, remove_zip: bool) -> None:
    """Extract one COCO zip part."""
    zip_path = root / ZIP_NAMES[part]
    outputs = expected_outputs(root, part)
    if all(path.exists() for path in outputs) and not force:
        print(f"Skipping extraction; {part} already exists.")
        return
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip not found: {zip_path}")

    print(f"Extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    normalize_extracted_layout(root)
    if remove_zip:
        zip_path.unlink(missing_ok=True)


def normalize_extracted_layout(root: Path) -> None:
    """Handle archives that may extract with an extra root directory."""
    nested = root / "coco2014"
    if nested.exists():
        for child in nested.iterdir():
            target = root / child.name
            if not target.exists():
                child.replace(target)
        try:
            nested.rmdir()
        except OSError:
            pass


def check_ready(root: Path, split: str) -> bool:
    """Print readiness checks for one split."""
    ok = True
    print(f"Checking COCO2014 {split} under {rel(root)} ...")
    for part in required_parts(split):
        zip_ok = (root / ZIP_NAMES[part]).exists()
        print(f"- {ZIP_NAMES[part]}: {'FOUND' if zip_ok else 'MISSING'}")
        for output in expected_outputs(root, part):
            exists = output.exists()
            print(f"- {rel(output)}: {'FOUND' if exists else 'MISSING'}")
            ok = ok and exists
    if ok:
        print(f"COCO2014 {split} is ready.")
    else:
        print(f"COCO2014 {split} is not ready yet.")
    return ok


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Download and extract official COCO2014 Caption files.")
    parser.add_argument("--split", default="val", choices=["train", "val", "trainval"])
    parser.add_argument("--root", default="data/raw/coco2014")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--keep-zip", dest="keep_zip", action="store_true", default=True)
    parser.add_argument("--remove-zip", dest="keep_zip", action="store_false")
    return parser.parse_args()


def main() -> None:
    """Download, extract, or check official COCO2014 files."""
    args = parse_args()
    root = Path(args.root)
    print_space_hint(root, args.split)

    if args.check and not (args.download or args.extract or args.download_only):
        raise SystemExit(0 if check_ready(root, args.split) else 2)

    do_download = args.download or args.download_only
    do_extract = args.extract and not args.download_only
    if not do_download and not do_extract:
        print("Nothing to do. Use --download, --extract, --download-only, or --check.")
        check_ready(root, args.split)
        return

    for part in required_parts(args.split):
        if do_download:
            download_part(root, part, args.num_workers, args.force)
        if do_extract:
            extract_part(root, part, args.force, remove_zip=not args.keep_zip)

    if args.split in {"val", "trainval"}:
        print("Checking annotations ...")
    ready = check_ready(root, args.split)
    if not ready:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
