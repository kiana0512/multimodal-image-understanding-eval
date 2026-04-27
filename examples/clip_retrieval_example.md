# CLIP Retrieval Example

Status: requires optional `open_clip_torch` and real or toy images.

```bash
pip install open_clip_torch
python scripts/create_toy_placeholders.py
python scripts/extract_clip_features.py --config configs/clip_retrieval.yaml
python scripts/run_text_image_retrieval.py --config configs/clip_retrieval.yaml
```

The Top-K CSV is a demo retrieval artifact, not a reported benchmark.
