"""AIGC quality scoring pipeline."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from mm_eval.evaluation.aigc_quality_metrics import blur_score, brightness_contrast_scores, resolution_check, aspect_ratio_check, make_badcase_reason, compute_overall_score

class AIGCQualityScorer:
    """Compute heuristic AIGC quality scores for a manifest."""
    def __init__(self, image_column: str = "image_path", prompt_column: str = "prompt"):
        self.image_column=image_column; self.prompt_column=prompt_column
    def score_manifest(self, manifest: str | Path, output_csv: str | Path, clip_scores: list[float] | None = None) -> pd.DataFrame:
        """Score all images and save CSV."""
        df=pd.read_csv(manifest); rows=[]
        for i,row in df.iterrows():
            image=row[self.image_column]
            bright,contrast=brightness_contrast_scores(image)
            item={**row.to_dict(),"clip_score": float(clip_scores[i]) if clip_scores is not None else 0.0,"blur_score": blur_score(image),"brightness_score": bright,"contrast_score": contrast,"duplicate_score": 0.0,"resolution_score": resolution_check(image),"aspect_ratio_score": aspect_ratio_check(image)}
            item["overall_score"]=compute_overall_score(pd.Series(item)); item["badcase_reason"]=make_badcase_reason(pd.Series(item)); rows.append(item)
        out=pd.DataFrame(rows); p=Path(output_csv); p.parent.mkdir(parents=True,exist_ok=True); out.to_csv(p,index=False); return out
