# Multimodal Image Understanding Eval

一个用于图文检索、CV 分割质量评估和 AIGC 图片质量评估的小型实验仓库。核心流程是：准备 manifest -> 跑脚本 -> 生成 `outputs/` 结果。

## 默认实验线

### 1. 图文检索 / CLIP / Top-K

默认数据集使用 HuggingFace Parquet 数据集：

```text
lambda/naruto-blip-captions
```

它包含真实图片和 caption，不依赖旧式 HF dataset script，也不需要额外从 COCO URL 下载图片。

```powershell
python scripts/prepare_caption_dataset.py --source hf --dataset lambda/naruto-blip-captions --max-samples 500 --force
python scripts/run_pipeline.py --task caption_retrieval
```

输出：

```text
data/processed/caption/caption_manifest.csv
outputs/features/caption_image_features.npy
outputs/features/caption_text_features.npy
outputs/features/caption_metadata.csv
outputs/retrieval/caption_text_to_image_topk.csv
outputs/retrieval/caption_retrieval_metrics.json
outputs/figures/caption_topk_contact_sheet.png
outputs/reports/experiment_report_*.md
```

### 2. CV 分类 / 分割伪标签质量评估

默认数据集继续使用 Oxford-IIIT Pet。

```powershell
python scripts/prepare_oxford_pet.py --source auto --max-samples 500 --make-pseudo-masks
python scripts/run_pipeline.py --task oxford_pet_segmentation
```

用于分类 baseline、GT mask、模拟 pseudo mask、Dice / IoU / Precision / Recall 分析。

### 3. AIGC 图片质量评估

默认不再下载 DiffusionDB 图片。图片质量评估推荐使用本地 ComfyUI / Stable Diffusion 输出目录。

```powershell
python scripts/prepare_local_game_assets.py --image-root D:/RT/game-aigc-asset-workflow/outputs --output data/processed/game_assets/game_asset_manifest.csv
python scripts/run_pipeline.py --task game_asset_quality
```

### 4. DiffusionDB prompt metadata 分析

DiffusionDB 2M / Large 体量非常大，而且当前 `datasets>=4.0` 对旧式 `diffusiondb.py` loading script 不兼容，所以默认只做 metadata-only prompt 分析。

```powershell
python scripts/prepare_diffusiondb_subset.py --mode metadata-only --max-samples 1000
python scripts/analyze_prompts.py --manifest data/processed/diffusiondb/diffusiondb_metadata_manifest.csv --text-column prompt
```

输出：

```text
data/processed/diffusiondb/diffusiondb_metadata_manifest.csv
outputs/prompt_analysis/prompt_stats.csv
outputs/prompt_analysis/prompt_word_freq.csv
outputs/reports/prompt_analysis_report_*.md
```

## 可选 / Legacy 数据

这些保留但不作为默认主线：

- `modelscope/coco_captions_small_slice`
- `modelscope/coco_2014_caption`
- `nlphuji/flickr30k`
- `poloclub/diffusiondb`

为什么 `coco_captions_small_slice` 只生成少量图片？它是 ModelScope 小样本数据，实际下载的是 CSV 和 metadata，图片字段指向 COCO URL。如果 `images.cocodataset.org` SSL / timeout 失败，就只能 materialize 很少图片。它适合测试字段解析，不适合正式 CLIP Top-K 检索。

如果一定要测试：

```powershell
python scripts/prepare_caption_dataset.py --source modelscope --dataset coco_captions_small_slice --max-samples 500
```

为什么不默认使用 DiffusionDB 图片？DiffusionDB 2M 约 TB 级，Large 更大；同时旧式 HF script 在新版 `datasets` 下不兼容。图片质量评估请使用本地生成图，或者手动准备 DiffusionDB 小图片子集后：

```powershell
python scripts/prepare_diffusiondb_subset.py --mode local-images --local-root data/raw/diffusiondb_sample --max-samples 100
python scripts/run_pipeline.py --task diffusiondb_quality
```

## 目录作用

- `configs/`：实验配置。
- `scripts/`：命令行脚本。
- `src/mm_eval/`：核心 Python 包。
- `data/`：示例、原始数据和处理后 manifest。
- `outputs/`：运行产物。
- `tests/`：单元测试。

## 常用诊断

```powershell
python scripts/check_env.py
python scripts/data_doctor.py
python -m pytest
```

## 结果怎么看

图文检索看：

- `outputs/retrieval/caption_text_to_image_topk.csv`
- `outputs/retrieval/caption_retrieval_metrics.json`
- `outputs/figures/caption_topk_contact_sheet.png`

重点指标：`recall_at_1`、`recall_at_5`、`recall_at_10`、`mrr`、`mean_rank`。

分割评估看：

- `outputs/segmentation_eval/*mask_metrics.csv`
- `outputs/segmentation_eval/*mask_metrics_summary.csv`

重点指标：`dice`、`iou`、`precision`、`recall`、`f1`。

AIGC 质量看：

- `outputs/quality_eval/*quality_scores.csv`
- `outputs/quality_eval/*badcases.csv`

重点字段：`overall_score`、`blur_score`、`brightness_score`、`contrast_score`、`resolution_score`、`badcase_reason`。

## 不要提交到 GitHub

```text
data/raw/
data/processed/
data/cache/
outputs/
*.npy
*.pth
```
