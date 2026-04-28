check:
	python scripts/check_env.py

test:
	python -m pytest

caption:
	python scripts/download_coco2014_official.py --split val --download --extract
	python scripts/prepare_coco2014_caption.py --split val --max-samples 5000 --force
	python scripts/run_pipeline.py --task caption_retrieval

oxford:
	python scripts/prepare_oxford_pet.py --source auto --max-samples 500 --make-pseudo-masks --force
	python scripts/run_pipeline.py --task oxford_pet_segmentation

aigc:
	python scripts/prepare_local_game_assets.py --image-root D:/your_aigc_outputs --output data/processed/game_assets/game_asset_manifest.csv
	python scripts/run_pipeline.py --task game_asset_quality
