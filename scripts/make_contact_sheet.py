from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import argparse
import pandas as pd
from mm_eval.visualization.contact_sheet import make_contact_sheet


def rows_from_retrieval_csv(path: str, num_queries: int, top_k: int) -> tuple[list[str], list[str]]:
    """Select Top-K retrieved images for the first N text queries."""
    df = pd.read_csv(path).dropna(subset=["image_path"])
    if "query_id" not in df.columns or "rank" not in df.columns:
        raise ValueError("Retrieval CSV must contain query_id, rank and image_path columns.")
    selected_queries = df["query_id"].drop_duplicates().head(num_queries).tolist()
    df = df[df["query_id"].isin(selected_queries)].sort_values(["query_id", "rank"])
    df = df.groupby("query_id", sort=False).head(top_k)
    captions = []
    for _, row in df.iterrows():
        marker = "+" if int(row.get("is_positive", 0)) == 1 else "-"
        query = str(row.get("query_text", ""))[:36]
        captions.append(f"{marker} q{row.get('query_id')} r{row.get('rank')}: {query}")
    return df["image_path"].astype(str).tolist(), captions


def main() -> None:
    """Create a contact sheet from CSV image paths."""
    parser=argparse.ArgumentParser(description="Make contact sheet from a CSV column, retrieval CSV, or image list.")
    parser.add_argument("--csv")
    parser.add_argument("--retrieval-csv")
    parser.add_argument("--image-column",default="image_path")
    parser.add_argument("--caption-column")
    parser.add_argument("--max-images",type=int,default=24)
    parser.add_argument("--num-queries",type=int,default=8)
    parser.add_argument("--top-k",type=int,default=5)
    parser.add_argument("--images",nargs="*")
    parser.add_argument("--output",default="outputs/figures/contact_sheet_demo.png")
    args=parser.parse_args()
    paths=args.images or []; captions=None
    if args.retrieval_csv:
        paths, captions = rows_from_retrieval_csv(args.retrieval_csv, args.num_queries, args.top_k)
    elif args.csv:
        df=pd.read_csv(args.csv).dropna(subset=[args.image_column])
        df=df.drop_duplicates(subset=[args.image_column]).head(args.max_images)
        paths=df[args.image_column].astype(str).tolist()
        if args.caption_column and args.caption_column in df.columns:
            captions=df[args.caption_column].fillna("").astype(str).tolist()
    make_contact_sheet(paths,args.output,captions=captions); print(args.output)
if __name__ == "__main__": main()
