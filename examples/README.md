# Examples

These examples explain command flows. They do not claim real benchmark results.

1. Create toy placeholders: `python scripts/create_toy_placeholders.py`
2. Build/check manifest: `python scripts/build_manifest.py --input data/sample_manifest.csv --output outputs/clean_manifest.csv --drop-missing`
3. Run AIGC heuristic scoring: `python scripts/run_aigc_quality_eval.py --config configs/aigc_quality_eval.yaml`
4. Generate report: `python scripts/generate_report.py --quality-csv outputs/quality_eval/aigc_quality_scores.csv`
