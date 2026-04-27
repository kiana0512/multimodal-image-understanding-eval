import numpy as np
from mm_eval.evaluation.segmentation_metrics import dice_score, iou_score, precision_recall_f1

def test_dice_score():
    gt=np.array([[1,1],[0,0]])
    pred=np.array([[1,0],[1,0]])
    assert abs(dice_score(gt,pred) - 0.5) < 1e-6

def test_iou_score():
    gt=np.array([[1,1],[0,0]])
    pred=np.array([[1,0],[1,0]])
    assert abs(iou_score(gt,pred) - (1/3)) < 1e-6

def test_precision_recall():
    gt=np.array([[1,1],[0,0]])
    pred=np.array([[1,0],[1,0]])
    precision,recall,f1=precision_recall_f1(gt,pred)
    assert abs(precision - 0.5) < 1e-6
    assert abs(recall - 0.5) < 1e-6
    assert abs(f1 - 0.5) < 1e-6
