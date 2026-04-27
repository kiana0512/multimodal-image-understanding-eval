$ErrorActionPreference = "Stop"
Write-Host "Running toy AIGC quality demo..."
python scripts/create_toy_placeholders.py
python scripts/run_aigc_quality_eval.py --config configs/aigc_quality_eval.yaml
python scripts/generate_report.py --quality-csv outputs/quality_eval/aigc_quality_scores.csv
Write-Host "Toy demo finished. Check outputs/quality_eval and outputs/reports."
