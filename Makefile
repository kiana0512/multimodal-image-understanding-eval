check:
	python scripts/check_env.py

test:
	python -m pytest

toy:
	python scripts/create_toy_placeholders.py
	python scripts/run_aigc_quality_eval.py --config configs/aigc_quality_eval.yaml
	python scripts/generate_report.py --quality-csv outputs/quality_eval/aigc_quality_scores.csv

oxford:
	python scripts/prepare_oxford_pet.py --max-samples 100 --make-pseudo-masks
	python scripts/run_pipeline.py --task oxford_pet_segmentation
