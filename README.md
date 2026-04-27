# Multimodal Image Understanding Eval

A lightweight experimental platform for multimodal image understanding and AIGC image quality evaluation.

This repository is designed as a portfolio-ready, interview-friendly Python project for image-text matching, CLIP/VLM feature analysis, AIGC image quality screening, image retrieval, segmentation/pseudo-label quality evaluation, and lightweight CV baselines.

## Supported Real-Data Demos

1. Flickr30K / COCO Caption: image-text retrieval.
2. Oxford-IIIT Pet: classification and segmentation-quality evaluation.
3. DiffusionDB subset: AIGC prompt-image quality evaluation.
4. Local game asset outputs: ComfyUI / SD generated asset evaluation.

## Why This Repo

Real CV and AIGC workflows need more than a single demo. They need data manifests, preprocessing, batch inference, feature caching, retrieval, metrics, badcase mining, visualization, and reports. This repo provides a compact but extensible implementation of that workflow.

## Features

- Real dataset preparation scripts for Flickr30K, COCO Caption, Oxford-IIIT Pet, DiffusionDB subset and local game assets.
- CSV manifest loading, validation, image path checks and toy manifest generation.
- Optional OpenCLIP encoder wrapper for image/text feature extraction.
- Cosine similarity, pairwise similarity matrix, Top-K retrieval export and retrieval metrics.
- AIGC proxy quality metrics: CLIP score placeholder, blur, brightness, contrast, duplicate, resolution and aspect ratio checks.
- Segmentation metrics: Dice, IoU, Precision, Recall, F1 and simplified Boundary F1.
- Lightweight CV classification demo using torchvision backbones.
- Markdown report generation and contact sheet visualization.

## Pipeline

```text
Images / Prompts / Labels
        ?
Manifest Builder
        ?
Image & Text Preprocessing
        ?
CLIP / OpenCLIP Feature Extraction
        ?
Retrieval / Matching / Scoring
        ?
AIGC Quality Evaluation
        ?
Badcase Mining & Visualization
        ?
Markdown / HTML Report
```

## Installation

Recommended conda environment:

```bash
conda env create -f environment.yml
conda activate llm
python scripts/check_env.py
python -m pytest
```

Minimal pip setup is also possible:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start: Toy Demo

```bash
python scripts/create_toy_placeholders.py
python scripts/run_aigc_quality_eval.py --config configs/aigc_quality_eval.yaml
python scripts/generate_report.py --quality-csv outputs/quality_eval/aigc_quality_scores.csv
```

## Real-Data Commands

```bash
python scripts/prepare_flickr30k_hf.py --max-samples 200
python scripts/run_pipeline.py --task flickr30k_retrieval

python scripts/prepare_oxford_pet.py --max-samples 200 --make-pseudo-masks
python scripts/run_pipeline.py --task oxford_pet_segmentation

python scripts/prepare_diffusiondb_subset.py --max-samples 100
python scripts/run_pipeline.py --task diffusiondb_quality

python scripts/prepare_local_game_assets.py --image-root D:/your_comfyui_outputs --output data/processed/game_assets/game_asset_manifest.csv
python scripts/run_pipeline.py --task game_asset_quality
```

## Dataset Format

Image-text manifest:

```csv
image_path,text,label,split,source,image_id,caption_id
data/processed/flickr30k/images/000001.jpg,"a dog running on grass",image_caption,test,source,000001,0
```

AIGC manifest:

```csv
image_path,prompt,negative_prompt,style_tag,asset_type,seed,model_name,split,source
data/processed/diffusiondb/images/000001.jpg,"fantasy sword icon","",unknown,ui_icon,1234,unknown,sample,hf
```

Segmentation manifest:

```csv
image_path,gt_mask_path,pred_mask_path,split,dataset
data/processed/oxford_pet/images/000001.jpg,data/processed/oxford_pet/masks/000001_gt.png,data/processed/oxford_pet/pseudo_masks/000001_pseudo.png,trainval,oxford_pet
```

## Output Files

- `outputs/retrieval/flickr30k_text_to_image_topk.csv`
- `outputs/quality_eval/diffusiondb_aigc_quality_scores.csv`
- `outputs/quality_eval/game_asset_quality_scores.csv`
- `outputs/segmentation_eval/oxford_pet_mask_metrics.csv`
- `outputs/reports/experiment_report_YYYYMMDD_HHMMSS.md`

## Roadmap

- Add human quality labels for AIGC score calibration.
- Add detection and instance segmentation evaluation.
- Add FAISS for larger-scale retrieval.
- Add Gradio UI for demo presentation.
- Extend game assets to multi-view 3D rendered images.

## Disclaimer

This repository is an experimental platform. It does not include large-scale private datasets, commercial model weights, or fake benchmark results. Generated outputs, toy images and simulated pseudo masks are `placeholder`, `demo result`, `not yet run`, or `example output` unless explicitly backed by real experiment logs.
