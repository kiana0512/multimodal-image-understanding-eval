# 实验结果分析

本文档总结当前项目已经跑通的两条真实数据集评估结果：

- `COCO2014 Caption`：用于 CLIP 图文检索，评估 text-to-image retrieval 的 Top-K 命中能力。
- `Oxford-IIIT Pet`：用于分割伪标签质量评估，比较 GT mask 与模拟 pseudo mask 的像素级一致性。

项目另保留本地 AIGC 资产质量评估流程，但它依赖用户自己的生成图目录，不属于本文档中的两个公开数据集结果。

## 0. 多模态评估框架说明

本项目中的多模态能力主要体现在三层：

1. 数据层：同时处理文本、图像、mask、prompt metadata 等不同模态或结构化信息。
2. 模型层：COCO2014 图文检索使用 CLIP / OpenCLIP 这类视觉语言双塔模型，把文本和图片编码到同一个 embedding 空间。
3. 评估层：不同任务使用不同指标，图文检索看 Recall@K / MRR，分割质量看 Dice / IoU / Boundary F1，本地 AIGC 资产看质量分数和 badcase。

严格来说，当前 COCO2014 流程使用的是 CLIP / OpenCLIP 视觉语言编码器，而不是生成式 VLM。它的多模态能力体现在“跨模态对齐”：文本 caption 和图片被编码成可比较的向量，系统可以用一段自然语言去检索对应图片。

COCO 检索链路如下：

```text
文本 caption query
  -> CLIP text encoder
  -> text embedding

COCO 图片 gallery
  -> CLIP image encoder
  -> image embedding

text embedding x image embedding
  -> cosine similarity
  -> Top-K 图片排序
  -> image_id 正样本判断
  -> Recall@K / MRR / mean_rank
```

Oxford 分割链路则是另一类多模态图像理解评估：样本同时包含原图、GT mask、pred mask。这里不做文本-图像对齐，而是评估像素级结构预测是否贴合真实目标区域。

## 1. 实验产物概览

### COCO2014 Caption 图文检索

本次 COCO2014 实验使用 `val2014` 子集构建检索任务：

```text
data/processed/caption/caption_manifest.csv
outputs/features/caption_image_features.npy
outputs/features/caption_text_features.npy
outputs/features/caption_metadata.csv
outputs/retrieval/caption_text_to_image_topk.csv
outputs/retrieval/caption_retrieval_metrics.json
outputs/figures/caption_topk_contact_sheet.png
```

本次评估规模：

| 项目 | 数值 |
| --- | ---: |
| caption query 数 | 5000 |
| gallery 图片数 | 4734 |
| Top-K 输出行数 | 50000 |
| 每个 query 返回候选数 | 10 |
| 模型 | OpenCLIP `ViT-B-32` |
| 预训练权重 | `laion2b_s34b_b79k` |

这里的 query 是 caption 文本，gallery 是去重后的图片集合。由于 COCO 每张图片有多条 caption，项目按 `image_id` 对 gallery 去重，避免同一张图片被重复编码；正样本判断使用 `query image_id == matched_image_id`，不使用 dataframe 行号。

这个设计很关键。COCO caption 是“一张图片对应多条人工描述”，如果直接按 CSV 行号算正样本，就会把同一张图片的多条 caption 误当成多张 gallery 图片，导致图片特征重复、指标偏差。当前实现中，图片侧按 `image_id` 去重，文本侧保留每条 caption，因此评估的是标准 text-to-image retrieval：给定一条文本描述，能否找回它对应的图片。

### Oxford-IIIT Pet 分割评估

Oxford 实验使用真实宠物图片和官方 trimap 构建 GT mask，并生成模拟 pseudo mask 来验证分割质量评估流程：

```text
data/processed/oxford_pet/oxford_pet_segmentation_manifest.csv
outputs/segmentation_eval/oxford_pet_mask_metrics.csv
outputs/segmentation_eval/oxford_pet_mask_metrics_summary.csv
```

本次评估规模：

| 项目 | 数值 |
| --- | ---: |
| 评估样本数 | 500 |
| split | `trainval` |
| GT 来源 | Oxford-IIIT Pet trimap |
| pred 来源 | 模拟 pseudo mask |
| 评估粒度 | 单张图片 mask pair |

需要注意：这里的 pseudo mask 是为了验证评估链路而构造的规则扰动结果，不代表某个真实分割模型的最终能力。它的价值在于证明 mask manifest、读取、二值化、指标计算、汇总报告这条流程可以稳定工作。

## 2. COCO2014 图文检索结果

### 指标定义

图文检索的目标是：输入一条 caption，系统从 gallery 图片集合中按相似度排序，返回最相关的图片。指标都围绕“正确图片排在第几名”展开。

| 指标 | 具体定义 | 直观理解 |
| --- | --- | --- |
| Top-1 / `recall_at_1` | 只看排名第 1 的图片，若 `matched_image_id == image_id` 则命中 | 文本一搜，第一张图就是正确图的比例 |
| `recall_at_5` | 看排名前 5 的图片，只要有一张是正确图就命中 | 前 5 个候选中能不能覆盖正确答案 |
| `recall_at_10` | 看排名前 10 的图片，只要有一张是正确图就命中 | 更宽松的候选召回能力 |
| `mrr` | Mean Reciprocal Rank，正确图片排名倒数的平均值 | 正确答案越靠前，分数越高 |
| `mean_rank` | 正确图片排名的平均值 | 越低越好，表示正确图整体排得更靠前 |

举例：如果某条 caption 的正确图片排第 1，那么它对 Recall@1、Recall@5、Recall@10 都贡献命中，MRR 贡献 `1/1=1`。如果正确图片排第 4，那么 Recall@1 不命中，Recall@5 和 Recall@10 命中，MRR 贡献 `1/4=0.25`。如果正确图片排第 12，那么 Recall@10 不命中，MRR 贡献 `1/12≈0.0833`。

Top-K CSV 中的 `score` 是文本向量与图片向量的 cosine similarity。分数越高，说明 CLIP embedding 空间认为这段文本和这张图片越匹配。`is_positive` 不是由分数决定，而是由 COCO 的真实 `image_id` 对齐关系决定。

### 核心指标

当前 `caption_retrieval_metrics.json` 结果如下：

| 指标 | 数值 | 含义 |
| --- | ---: | --- |
| `recall_at_1` | 0.3976 | 39.76% 的 caption 能在第 1 名直接找回对应图片 |
| `recall_at_5` | 0.6616 | 66.16% 的 caption 能在前 5 名内找回对应图片 |
| `recall_at_10` | 0.7570 | 75.70% 的 caption 能在前 10 名内找回对应图片 |
| `mrr` | 0.5200 | 正确图片平均排名质量较好，越接近 1 越好 |
| `mean_rank` | 13.4604 | 正确图片平均出现在第 13.46 名 |
| `top1_accuracy` | 0.3976 | 与 Recall@1 等价 |

从结果看，CLIP 在 COCO caption 到 image 的检索上表现稳定：约四成 query 可以第一名命中，约三分之二 query 可以在前五名命中，超过四分之三 query 可以在前十名命中。

这里的 Top-1 不是分类准确率，而是跨模态检索的第一候选命中率。它回答的问题是：“给一句 caption，让模型从 4734 张图片里选，第一张是不是对应图片？”因此 `recall_at_1=0.3976` 代表在 5000 条文本 query 中，有 1988 条的第一张候选图就是正确图。

### Top-K 命中分布

根据 `caption_text_to_image_topk.csv` 聚合：

| Top-K 范围 | 命中 query 数 | 占比 |
| --- | ---: | ---: |
| Top-1 | 1988 / 5000 | 39.76% |
| Top-5 | 3308 / 5000 | 66.16% |
| Top-10 | 3785 / 5000 | 75.70% |

这个分布说明：模型并不是只在少数简单样本上工作，而是在较大比例的 query 上能把正确图片排到候选列表前部。Top-1 到 Top-5 的提升为 26.40 个百分点，说明很多 query 的正确图片已经在相近语义候选里，只是第一名可能被视觉或文本语义非常接近的图片替代。

### 相似度分数观察

Top-K CSV 中的分数是文本特征与图片特征的 cosine similarity。当前聚合统计为：

| 项目 | 平均分 |
| --- | ---: |
| Top-1 候选平均分 | 0.3317 |
| 正样本候选平均分 | 0.3233 |
| 负样本候选平均分 | 0.2899 |

正样本平均分高于负样本，说明 embedding 空间确实学到了跨模态对齐关系。但 Top-1 候选平均分高于正样本平均分也很正常：Top-1 永远是模型认为最相似的候选，其中一部分是语义非常接近但不是同一张图的 hard negative。

例如 query `A cat laying in a shoe on the ground.` 的 Top-1 是另一张“猫和鞋”相关图片，Top-2 才是正确图片。这类错误不是完全无关的误检，而是语义相近图像之间的排序问题。对于展示型检索系统，这说明 Top-K 结果往往仍然有可解释性；对于严格 image_id 级 benchmark，则会被计为 Top-1 miss。

这也体现了 VLM / CLIP 检索的特点：模型并不是用固定类别标签做分类，而是在开放语义空间中计算相似度。因此它可能把“猫躺在鞋旁边”和“猫把头伸进鞋里”排得很近。这种结果从用户检索体验看通常是相关的，但从 COCO benchmark 的 image_id 级评估看，只有原图才算正样本。

### 结果意义

COCO2014 结果证明了项目已经具备完整的真实图文检索能力：

- 可以从官方 COCO zip 构建标准 manifest。
- 可以正确处理“一张图多条 caption”的数据结构。
- 可以避免重复编码 gallery 图片。
- 可以用 CLIP / OpenCLIP 提取 image/text feature。
- 可以输出可审计的 Top-K CSV，包括 query、rank、matched image、score、is_positive。
- 可以基于 `image_id` 计算 Recall@K、MRR、mean rank。
- 可以生成 contact sheet，支持直观看 Top-K 检索质量。

整体上，这条实验线已经从 toy demo 变成了可复现、可解释、可扩展的真实数据评估流程。

## 3. Oxford-IIIT Pet 分割结果

### 指标定义

Oxford 分割任务不是检索任务，而是 mask 质量评估任务。它比较的是 `gt_mask_path` 和 `pred_mask_path` 两张二值 mask：

| 指标 | 具体定义 | 直观理解 |
| --- | --- | --- |
| `dice` | `2 * intersection / (gt_area + pred_area)` | 两个 mask 的整体重叠程度 |
| `iou` | `intersection / union` | 交集占并集比例，比 Dice 更严格 |
| `precision` | `TP / (TP + FP)` | pred 里有多少是真的前景，低值表示过分割 |
| `recall` | `TP / (TP + FN)` | GT 前景有多少被找回，低值表示欠分割 |
| `f1` | precision 和 recall 的调和平均 | 综合衡量像素级准确性 |
| `boundary_f1_simple` | 基于 mask 边界梯度的简化 F1 | 关注轮廓边缘是否对齐 |

Dice 和 IoU 看区域重叠，Boundary F1 看边界。一个 mask 可能主体区域很准，所以 Dice / IoU 很高，但边缘略有膨胀、腐蚀或平移，Boundary F1 就会明显下降。

### 核心指标

当前 `oxford_pet_mask_metrics_summary.csv` 结果如下：

| 指标 | mean | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| `dice` | 0.9764 | 0.0275 | 0.7724 | 1.0000 |
| `iou` | 0.9553 | 0.0495 | 0.6292 | 1.0000 |
| `precision` | 0.9785 | 0.0331 | 0.7724 | 1.0000 |
| `recall` | 0.9752 | 0.0339 | 0.7724 | 1.0000 |
| `f1` | 0.9764 | 0.0275 | 0.7724 | 1.0000 |
| `boundary_f1_simple` | 0.6393 | 0.2839 | 0.0675 | 1.0000 |

从像素区域重叠指标看，pseudo mask 与 GT mask 的整体一致性很高：平均 Dice 达到 0.9764，平均 IoU 达到 0.9553。Precision 与 Recall 也比较均衡，分别为 0.9785 和 0.9752，说明没有明显的系统性过分割或欠分割倾向。

### 样本覆盖情况

根据单样本 metrics 聚合：

| 条件 | 样本占比 |
| --- | ---: |
| `dice >= 0.95` | 87.2% |
| `iou >= 0.90` | 87.6% |
| `boundary_f1_simple >= 0.80` | 40.2% |

区域重叠指标明显优于边界指标。原因是 Dice / IoU 更关注前景区域整体重叠，只要主体区域大部分对齐就能获得较高分；`boundary_f1_simple` 对轮廓边缘更敏感，轻微腐蚀、膨胀、平移或模糊阈值都会明显降低边界分。

### 最低分样本观察

当前 Dice 最低的几个样本包括：

| image_path | dice | iou | precision | recall | boundary_f1_simple |
| --- | ---: | ---: | ---: | ---: | ---: |
| `data/processed/oxford_pet/images/000248_Beagle.jpg` | 0.7724 | 0.6292 | 0.7724 | 0.7724 | 0.3363 |
| `data/processed/oxford_pet/images/000243_Beagle.jpg` | 0.8134 | 0.6854 | 0.8134 | 0.8134 | 0.1428 |
| `data/processed/oxford_pet/images/000363_Bombay.jpg` | 0.8642 | 0.7609 | 0.8642 | 0.8642 | 0.1506 |
| `data/processed/oxford_pet/images/000423_Boxer.jpg` | 0.8655 | 0.7629 | 0.8655 | 0.8655 | 0.3041 |
| `data/processed/oxford_pet/images/000128_American_Pit_Bull_Terrier.jpg` | 0.8676 | 0.7662 | 0.8676 | 0.8676 | 0.3000 |

这些 badcase 的意义在于：即使整体平均指标很高，单样本层面仍然能暴露边界错位、局部缺失、局部扩张等质量问题。因此本项目不只输出 summary，也保留了每张图的指标 CSV，便于后续做 badcase mining。

### 结果意义

Oxford 分割实验证明项目已经具备以下能力：

- 从真实 Oxford-IIIT Pet 数据构建 image / GT mask / pred mask manifest。
- 将 Oxford trimap 转换为二值前景 mask。
- 批量计算 Dice、IoU、Precision、Recall、F1、Boundary F1。
- 同时输出单样本明细和全局 summary。
- 支持定位最差样本，用于后续可视化、人工复核或模型迭代。

这条实验线的重点不是证明模拟 pseudo mask 很强，而是证明评估系统可以处理真实图像和真实 GT mask，并稳定产出可解释的分割质量指标。

## 4. 两条数据线的互补性

COCO2014 和 Oxford-IIIT Pet 覆盖了不同类型的多模态图像理解能力：

| 数据线 | 评估对象 | 主要能力 | 关键输出 |
| --- | --- | --- | --- |
| COCO2014 Caption | 文本到图片检索 | 跨模态语义对齐 | Recall@K、MRR、Top-K CSV、contact sheet |
| Oxford-IIIT Pet | mask pair 质量 | 像素级分割一致性 | Dice、IoU、Precision、Recall、Boundary F1 |

COCO 检索更关注“文本描述能否找到正确图片”，体现的是全局语义匹配能力；Oxford 分割更关注“预测 mask 是否贴合目标区域”，体现的是局部像素级质量控制能力。两者结合，可以展示项目不只是跑一个模型 demo，而是把数据准备、特征提取、评估指标、可视化和报告串成了完整实验平台。

从多模态角度看，COCO2014 展示的是语言和视觉之间的对齐能力；Oxford 展示的是视觉结构和像素级标注之间的一致性评估能力。前者回答“文字和图片是否语义匹配”，后者回答“模型或伪标签是否准确定位图像中的目标区域”。这两类能力结合起来，构成了多模态图像理解评估中非常常见的两端：语义级理解和像素级质量控制。

## 5. 当前结果的边界和后续方向

当前结果已经足够说明流程完整，但仍有明确提升空间：

- COCO 当前使用 `val2014` 的 5000 条 caption pair；如果需要更稳定的统计，可以扩大到 `trainval` 的 20000 条或更多。
- 当前检索模型使用 `ViT-B-32`，可以对比更大的 OpenCLIP backbone，例如 `ViT-L-14`，观察 Recall@K 提升。
- Contact sheet 可以进一步加入 query 分组、正样本高亮、hard negative 展示。
- Oxford 当前 pred mask 是模拟扰动结果；后续可以接入真实分割模型输出，例如 SAM、MedSAM 或自训练模型。
- Boundary F1 明显低于区域指标，后续可以增加边界可视化，专门分析轮廓错位问题。

## 6. 总结

当前项目已经跑通两条真实数据集验证流程：

1. COCO2014 Caption 检索在 5000 个 caption query、4734 张 gallery 图上达到 `Recall@1=39.76%`、`Recall@5=66.16%`、`Recall@10=75.70%`、`MRR=0.5200`。
2. Oxford-IIIT Pet 分割评估在 500 张样本上达到平均 `Dice=0.9764`、`IoU=0.9553`、`F1=0.9764`。

这说明项目已经从数据下载、manifest 构建、模型推理、指标计算、结果导出到可视化报告形成了闭环。COCO2014 展示跨模态语义检索能力，Oxford-IIIT Pet 展示像素级分割质量评估能力，两者共同构成了一个可扩展的多模态图像理解评估框架。
