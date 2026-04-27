from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def check_import(module_name: str, display_name: str | None = None) -> tuple[bool, str]:
    """Try importing a module and return status plus version when available."""
    display = display_name or module_name
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", "version unknown")
        return True, f"[OK] {display}: {version}"
    except Exception as exc:
        return False, f"[MISS] {display}: {exc}"


def main() -> None:
    """Print environment diagnostics for the llm conda environment."""
    print("== Environment Check ==")
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    print(f"Working directory: {Path.cwd()}")

    ok, msg = check_import("torch")
    print(msg)
    if ok:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device 0: {torch.cuda.get_device_name(0)}")

    for module, name in [
        ("torchvision", "torchvision"),
        ("open_clip", "open_clip"),
        ("datasets", "datasets"),
        ("modelscope", "modelscope"),
        ("mm_eval", "mm_eval"),
    ]:
        _, msg = check_import(module, name)
        print(msg)

    print("\nIf optional packages are missing, install them only when the related dataset/model flow is needed.")


if __name__ == "__main__":
    main()
