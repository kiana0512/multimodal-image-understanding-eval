# AIGC Game Asset Evaluation Example

Status: example output / not yet run on real game production assets.

```bash
python scripts/create_toy_placeholders.py
python scripts/run_aigc_quality_eval.py --config configs/aigc_quality_eval.yaml
```

Expected outputs include `outputs/quality_eval/aigc_quality_scores.csv` and `outputs/quality_eval/badcases.csv`.
