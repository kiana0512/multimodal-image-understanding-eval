# Segmentation Quality Evaluation Example

Prepare masks listed in `data/sample_segmentation_pairs.csv`, then run:

```bash
python scripts/run_segmentation_quality_eval.py --config configs/segmentation_quality_eval.yaml
```

Outputs: per-sample Dice, IoU, Precision, Recall, F1 and a simplified Boundary F1.
