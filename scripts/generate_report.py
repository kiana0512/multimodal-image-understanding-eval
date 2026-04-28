from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
from mm_eval.evaluation.report_builder import build_markdown_report, save_markdown_report

def main() -> None:
    """Generate markdown experiment report."""
    parser=argparse.ArgumentParser(description="Generate markdown report from CSV outputs.")
    parser.add_argument("--retrieval-csv"); parser.add_argument("--retrieval-metrics-json"); parser.add_argument("--quality-csv"); parser.add_argument("--segmentation-csv"); parser.add_argument("--config"); parser.add_argument("--output-dir",default="outputs/reports")
    args=parser.parse_args(); md=build_markdown_report(retrieval_csv=args.retrieval_csv,retrieval_metrics_json=args.retrieval_metrics_json,quality_csv=args.quality_csv,segmentation_csv=args.segmentation_csv,config_path=args.config); path=save_markdown_report(md,args.output_dir); print(path)
if __name__ == "__main__": main()
