from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from mm_eval.utils.config import load_config
from mm_eval.cv.classification_train import run_classification_demo

def main() -> None:
    """Run lightweight CV classification baseline."""
    parser=argparse.ArgumentParser(description="Run a small classification training demo.")
    parser.add_argument("--config", required=True)
    args=parser.parse_args(); metrics=run_classification_demo(load_config(args.config)); print(metrics)
if __name__ == "__main__": main()
