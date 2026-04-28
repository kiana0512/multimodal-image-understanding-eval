# Multimodal Image Understanding Eval

一个面向真实图像数据的多模态评估项目，覆盖图文检索、分割质量评估和本地 AIGC 资产质量评估。项目重点不是 toy demo，而是把数据下载、manifest 构建、模型推理、指标计算、Top-K 导出、可视化和报告生成串成可复现流程。

当前保留三条主线：

- 官方 COCO2014 Caption：图文检索、Top-K、Recall@K、MRR。
- Oxford-IIIT Pet：真实图片和 mask，用于分割伪标签质量评估。
- 本地 AIGC 资产：扫描你自己生成的 ComfyUI / Stable Diffusion / 游戏资产图片，做质量评分和 badcase 挖掘。

完整结果分析见：

```text
RESULTS_ANALYSIS.md
```

## 项目流程

```text
真实数据/本地资产
  -> prepare 脚本生成 manifest
  -> run_pipeline.py 执行评估
  -> outputs/ 输出特征、指标、Top-K、badcase 和报告
```

## 多模态能力如何体现

本项目的“多模态”不是只把图片文件和文本文件放在一起，而是分别处理不同模态的数据，并在统一评估接口下计算可解释指标：

| 模态组合 | 对应任务 | 项目中如何实现 |
| --- | --- | --- |
| 文本 + 图像 | COCO2014 图文检索 | caption 文本作为 query，图片作为 gallery，用 CLIP / OpenCLIP 编码到同一向量空间后做相似度检索 |
| 图像 + mask | Oxford-IIIT Pet 分割评估 | 原图、GT mask、pred mask 组成样本，计算像素级 Dice、IoU、Precision、Recall |
| 图像 + prompt metadata | 本地 AIGC 资产评估 | 生成图、prompt、style、asset type 等字段组成 manifest，输出质量分数和 badcase |

CLIP / OpenCLIP 在这里承担的是视觉-语言检索模型的角色，也可以理解为一类 VLM-style 的双塔视觉语言编码器：图片经过 image encoder，文本经过 text encoder，二者被映射到同一个 embedding 空间。检索时不需要额外分类器，而是直接比较文本向量和图片向量的 cosine similarity。

## 环境准备

```powershell
conda env create -f environment.yml
conda activate llm
python scripts/check_env.py
```

如果 `llm` 环境已经存在：

```powershell
conda activate llm
conda env update -f environment.yml
python scripts/check_env.py
```

下载脚本会优先使用 `aria2c` 多线程断点续传；如果没有 `aria2c`，会自动 fallback 到 Python 下载。建议安装：

```powershell
conda install -c conda-forge aria2 -y
```

## 目录结构

```text
configs/
  caption_clip_retrieval.yaml
  game_asset_quality_eval.yaml
  oxford_pet_segmentation_quality.yaml

scripts/
  download_coco2014_official.py
  prepare_coco2014_caption.py
  prepare_oxford_pet.py
  prepare_local_game_assets.py
  run_pipeline.py
  data_doctor.py

src/mm_eval/
  data/
  models/
  retrieval/
  evaluation/
  aigc/
  visualization/

data/
  raw/
  processed/

outputs/
  features/
  retrieval/
  segmentation_eval/
  quality_eval/
  figures/
  reports/
```

`data/` 和 `outputs/` 是运行产物目录，不应提交到 GitHub。

## 1. COCO2014 Caption 图文检索

默认数据集是官方 COCO2014 Caption：

```text
val2014.zip
train2014.zip
annotations_trainval2014.zip
```

官方 COCO2014 包含真实图片和人工 caption，每张图片有多条 caption，适合 CLIP text-to-image retrieval、Top-K、Recall@K、MRR 和 contact sheet 展示。

### CLIP / VLM 检索链路

COCO2014 图文检索的执行链路如下：

```text
captions_val2014.json + val2014 images
  -> prepare_coco2014_caption.py
  -> data/processed/caption/caption_manifest.csv
  -> extract_clip_features.py
     - image encoder 编码唯一图片 gallery
     - text encoder 编码每条 caption query
  -> run_text_image_retrieval.py
     - 计算 text feature 与 image feature 的 cosine similarity
     - 导出 Top-K CSV
     - 基于 image_id 判断正样本并计算 Recall@K / MRR
```

项目中 CLIP / VLM 检索主要体现在这些产物里：

- `outputs/features/caption_image_features.npy`：图片模态向量。
- `outputs/features/caption_text_features.npy`：文本模态向量。
- `outputs/retrieval/caption_text_to_image_topk.csv`：每条文本 query 检索到的 Top-K 图片结果。
- `outputs/retrieval/caption_retrieval_metrics.json`：跨模态检索指标。
- `outputs/figures/caption_topk_contact_sheet.png`：把检索结果可视化成图片网格。

### 数据下载检查

只检查文件是否已经下载和解压：

```powershell
python scripts/download_coco2014_official.py --split val --check
```

### 快速版：只跑 val2014

```powershell
python scripts/download_coco2014_official.py --split val --download --extract
python scripts/prepare_coco2014_caption.py --split val --max-samples 5000 --force
python scripts/run_pipeline.py --task caption_retrieval
```

### 完整版：跑 train + val

```powershell
python scripts/download_coco2014_official.py --split trainval --download --extract
python scripts/prepare_coco2014_caption.py --split trainval --max-samples 20000 --force
python scripts/run_pipeline.py --task caption_retrieval
```

### 磁盘空间

- `val2014`：约 6GB。
- `train2014`：约 13GB。
- `annotations_trainval2014`：约 241MB。
- 完整 train+val 解压后建议预留 30GB 以上。

### 输出文件

```text
data/raw/coco2014/val2014/
data/raw/coco2014/train2014/
data/raw/coco2014/annotations/captions_val2014.json
data/raw/coco2014/annotations/captions_train2014.json

data/processed/caption/caption_manifest.csv

outputs/features/caption_image_features.npy
outputs/features/caption_text_features.npy
outputs/features/caption_metadata.csv

outputs/retrieval/caption_text_to_image_topk.csv
outputs/retrieval/caption_retrieval_metrics.json

outputs/figures/caption_topk_contact_sheet.png
outputs/reports/experiment_report_*.md
```

### 结果怎么看

重点看：

- `outputs/retrieval/caption_text_to_image_topk.csv`
- `outputs/retrieval/caption_retrieval_metrics.json`
- `outputs/figures/caption_topk_contact_sheet.png`

核心指标：

- `recall_at_1`：也就是 Top-1 命中率。每条 caption 只看排名第 1 的图片，如果它的 `matched_image_id` 等于 query 对应的 `image_id`，就算命中。
- `recall_at_5`：每条 caption 看前 5 张候选图，只要其中任意一张是正确图片，就算命中。
- `recall_at_10`：每条 caption 看前 10 张候选图，只要其中任意一张是正确图片，就算命中。
- `mrr`：Mean Reciprocal Rank，正确图片排名的倒数均值。正确图片排第 1 贡献 1，排第 2 贡献 0.5，排第 10 贡献 0.1，越高越好。
- `mean_rank`：正确图片的平均排名，越低越好。

Top-K CSV 中的关键字段含义：

| 字段 | 含义 |
| --- | --- |
| `query_id` | 当前 caption query 的 ID |
| `query_text` | 输入检索系统的文本描述 |
| `rank` | 当前候选图片在该 query 下的排名 |
| `image_path` | 被检索出来的候选图片 |
| `score` | 文本向量与图片向量的 cosine similarity |
| `is_positive` | 是否为正确图片，判断条件是 `image_id == matched_image_id` |
| `image_id` | query caption 对应的真实图片 ID |
| `matched_image_id` | 当前候选图片的图片 ID |

当前一次已跑通结果：

| 指标 | 数值 |
| --- | ---: |
| query 数 | 5000 |
| gallery 图片数 | 4734 |
| `recall_at_1` | 0.3976 |
| `recall_at_5` | 0.6616 |
| `recall_at_10` | 0.7570 |
| `mrr` | 0.5200 |
| `mean_rank` | 13.4604 |

更详细的结果解释见 `RESULTS_ANALYSIS.md`。

## 2. Oxford-IIIT Pet 分割评估

自动下载或复用本地 Oxford-IIIT Pet：

```powershell
python scripts/prepare_oxford_pet.py --source auto --max-samples 500 --make-pseudo-masks --force
python scripts/run_pipeline.py --task oxford_pet_segmentation
```

如果已经手动解压到 `data/raw/oxford_pet`：

```powershell
python scripts/prepare_oxford_pet.py --source local --local-root data/raw/oxford_pet --max-samples 500 --make-pseudo-masks --force
python scripts/run_pipeline.py --task oxford_pet_segmentation
```

输出：

```text
data/processed/oxford_pet/oxford_pet_segmentation_manifest.csv
outputs/segmentation_eval/oxford_pet_mask_metrics.csv
outputs/segmentation_eval/oxford_pet_mask_metrics_summary.csv
```

重点指标：`dice`、`iou`、`precision`、`recall`、`f1`。

这些指标代表：

- `dice`：GT mask 和 pred mask 的重叠程度，越接近 1 说明整体区域越一致。
- `iou`：Intersection over Union，交集面积除以并集面积，越接近 1 越好。
- `precision`：pred mask 中有多少像素是真的前景，低 precision 通常表示过分割。
- `recall`：GT 前景中有多少被 pred 找回来，低 recall 通常表示欠分割。
- `f1`：precision 和 recall 的调和平均。
- `boundary_f1_simple`：简化边界 F1，更关注轮廓边缘是否对齐。

当前一次已跑通结果：

| 指标 | mean |
| --- | ---: |
| `dice` | 0.9764 |
| `iou` | 0.9553 |
| `precision` | 0.9785 |
| `recall` | 0.9752 |
| `f1` | 0.9764 |
| `boundary_f1_simple` | 0.6393 |

注意：当前 `pred_mask_path` 指向模拟 pseudo mask，主要用于验证分割评估链路。后续可以替换成真实分割模型的输出 mask。

## 3. 本地 AIGC 资产质量评估

这个流程不下载外部数据集，只扫描你自己生成的 AIGC 图片目录：

```powershell
python scripts/prepare_local_game_assets.py --image-root D:/your_aigc_outputs --output data/processed/game_assets/game_asset_manifest.csv
python scripts/run_pipeline.py --task game_asset_quality
```

如果有 prompt metadata，可以额外传：

```powershell
python scripts/prepare_local_game_assets.py --image-root D:/your_aigc_outputs --prompts-csv D:/your_aigc_outputs/prompts.csv --output data/processed/game_assets/game_asset_manifest.csv
python scripts/run_pipeline.py --task game_asset_quality
```

输出：

```text
data/processed/game_assets/game_asset_manifest.csv
outputs/quality_eval/game_asset_quality_scores.csv
outputs/quality_eval/game_asset_badcases.csv
outputs/quality_eval/game_asset_badcase_report.md
outputs/reports/experiment_report_*.md
```

重点字段：`overall_score`、`blur_score`、`brightness_score`、`contrast_score`、`resolution_score`、`badcase_reason`。

## Pipeline 任务

统一入口是 `scripts/run_pipeline.py`：

```powershell
python scripts/run_pipeline.py --task caption_retrieval
python scripts/run_pipeline.py --task oxford_pet_segmentation
python scripts/run_pipeline.py --task game_asset_quality
```

`run_pipeline.py` 不负责下载数据。它只在 manifest 已经存在时执行评估。如果 manifest 缺失，终端会打印对应的 prepare 命令。

## Manifest 约定

COCO2014 图文检索 manifest：

```csv
image_path,text,label,split,source,image_id,caption_id,dataset_name
```

Oxford 分割 manifest：

```csv
image_path,gt_mask_path,pred_mask_path,split,dataset
```

AIGC 资产 manifest：

```csv
image_path,asset_id,asset_type,style_tag,prompt,negative_prompt,seed,model_name,split,source
```

这些 manifest 是项目的核心数据接口。只要列名保持一致，就可以替换数据来源或模型输出。

## 保留范围

默认只维护 COCO2014、Oxford-IIIT Pet 和本地 AIGC 资产三条流程。其他旧数据源 wrapper、下载脚本和配置已经从项目主线中清理。

## 常用诊断

```powershell
python scripts/check_env.py
python scripts/data_doctor.py
python -m pytest
```

`data_doctor.py` 会检查 COCO2014、Oxford 和 AIGC 三条线的数据与输出是否存在，适合每次运行前后做状态确认。

## 常见问题

### COCO 下载很慢怎么办

优先安装 `aria2c`：

```powershell
conda install -c conda-forge aria2 -y
```

下载脚本会保留 zip，并支持中断后继续运行。

### 为什么 COCO gallery 图片数少于 caption 数

COCO 每张图片有多条 caption。检索时 query 是 caption 行，gallery 是按 `image_id` 去重后的图片集合，所以 caption 数会大于图片数。

### 为什么 Oxford 的 boundary 分数低于 Dice / IoU

Dice 和 IoU 看整体区域重叠，boundary F1 对边缘错位更敏感。轻微腐蚀、膨胀、平移或模糊阈值都会明显影响边界分。

### AIGC 流程是否下载外部数据集

不下载。AIGC 流程只扫描你本地生成的图片目录，适合评估自己生成的游戏资产、图标、角色图、场景图等。

## 不要提交到 GitHub

```text
data/raw/
data/processed/
data/cache/
outputs/
*.npy
*.pth
```
