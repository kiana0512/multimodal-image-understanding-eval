# Data Format

This repository does not include large datasets. CSV files here are schema examples and toy manifests only. Missing images should be created with `python scripts/create_toy_placeholders.py` or replaced with your own data.

## 1. General Image-Text Data

Used for CLIP/VLM matching, text-to-image retrieval, image-to-text retrieval and prompt-result alignment.

```csv
image_path,text,label,split,source
data/toy_images/cat_001.jpg,"a cute cat sitting on the sofa",cat,train,toy
```

## 2. AIGC Generated Images

Used for prompt consistency, style checks, badcase mining and batch filtering.

```csv
image_path,prompt,negative_prompt,style_tag,asset_type,seed,model_name,split
outputs/aigc/char_001.png,"anime girl, blue hair, game character","low quality",anime,character,1234,sdxl,val
```

## 3. Game Assets

Used for character concept art, UI icons, props, scene concepts and future 3D multi-view renders.

```csv
image_path,asset_id,asset_type,style_tag,text_desc,split
assets/icons/sword_001.png,sword_001,ui_icon,fantasy,"silver fantasy sword icon",val
```

## 4. Segmentation / Pseudo-label Quality

Used for SAM/MedSAM-style pseudo-label filtering and teacher-student diagnostics.

```csv
image_path,gt_mask_path,pred_mask_path,split,dataset
data/seg/images/001.png,data/seg/gt/001.png,data/seg/pred/001.png,val,woundseg_toy
```

## 5. Classification Demo

The classification module supports `torchvision` CIFAR10 and custom `ImageFolder`. This is a CV baseline, not the main contribution of the repo.
