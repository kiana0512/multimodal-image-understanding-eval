check:
	python scripts/check_env.py

test:
	python -m pytest

caption:
	python scripts/prepare_caption_dataset.py --source hf --dataset lambda/naruto-blip-captions --max-samples 500 --force
	python scripts/run_pipeline.py --task caption_retrieval

prompts:
	python scripts/prepare_diffusiondb_subset.py --mode metadata-only --max-samples 1000
	python scripts/analyze_prompts.py --manifest data/processed/diffusiondb/diffusiondb_metadata_manifest.csv --text-column prompt

oxford:
	python scripts/prepare_oxford_pet.py --source auto --max-samples 100 --make-pseudo-masks
	python scripts/run_pipeline.py --task oxford_pet_segmentation
