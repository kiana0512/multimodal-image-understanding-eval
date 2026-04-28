conda activate llm
python scripts/download_coco2014_official.py --split val --download --extract
python scripts/prepare_coco2014_caption.py --split val --max-samples 5000 --force
python scripts/run_pipeline.py --task caption_retrieval
