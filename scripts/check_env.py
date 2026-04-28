from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check_import(module_name: str, display_name: str | None = None, purpose: str | None = None) -> tuple[bool, str]:
    """Try importing a module and return status plus version when available."""
    display = display_name or module_name
    suffix = f" ({purpose})" if purpose else ""
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "version unknown")
        return True, f"[OK] {display}: {version}{suffix}"
    except Exception as exc:
        return False, f"[MISS] {display}: {exc}{suffix}"


def main() -> None:
    """Print environment diagnostics for the llm conda environment."""
    print("== Environment Check ==")
    version = sys.version_info
    status = "OK" if (version.major, version.minor) in {(3, 10), (3, 11)} else "WARNING"
    print("Recommended Python version: 3.10 or 3.11")
    print(f"Current Python: {sys.version.split()[0]}")
    print(f"Status: {status}")
    if version.major > 3 or (version.major == 3 and version.minor >= 12):
        print("Warning: Python 3.12+ may have compatibility risks with some CV/ML dependencies.")
    print(f"Platform: {platform.platform()}")
    print(f"Working directory: {Path.cwd()}")

    ok, msg = check_import("torch")
    print(msg)
    if ok:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")

    for module, name, purpose in [
        ("torchvision", "torchvision", "image transforms used by CLIP dependencies"),
        ("cv2", "opencv-python / cv2", "required for Oxford Pet mask IO and segmentation metrics"),
        ("open_clip", "open_clip", "required for CLIP retrieval feature extraction"),
        ("mm_eval", "mm_eval", "local project package"),
    ]:
        _, msg = check_import(module, name, purpose)
        print(msg)

    print("\nOfficial COCO2014 download uses urllib/aria2c and does not require HF or ModelScope packages.")


if __name__ == "__main__":
    main()
