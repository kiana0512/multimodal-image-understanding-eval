from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from mm_eval.utils.config import load_config
from mm_eval.retrieval.image_retrieval import compute_image_retrieval_scores
from mm_eval.retrieval.topk_export import export_topk

def main() -> None:
    """Run image-image retrieval from cached features."""
    parser=argparse.ArgumentParser(description="Run image-image retrieval and export Top-K CSV.")
    parser.add_argument("--config", required=True)
    args=parser.parse_args(); cfg=load_config(args.config); out=Path(cfg.get("output_dir","outputs/retrieval")); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(cfg["manifest_path"]); features=np.load(Path(cfg.get("feature_dir","outputs/features"))/"image_features.npy")
    scores=compute_image_retrieval_scores(features,features); ids=df[cfg.get("image_column","image_path")].astype(str).tolist(); export_topk(scores,ids,ids,out/"image_to_image_topk.csv",k=int(cfg.get("top_k",5)))
    print(f"Saved image retrieval results to {out}")
if __name__ == "__main__": main()
