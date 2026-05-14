from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], threshold: float) -> Dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int)
    y_score_arr = np.asarray(y_score, dtype=float)
    y_pred_arr = (y_score_arr > threshold).astype(int)

    if len(np.unique(y_true_arr)) < 2:
        return {
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
            "f1_score": float("nan"),
            "sensitivity": float("nan"),
            "specificity": float("nan"),
            "precision": float("nan"),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    return {
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, y_score_arr)),
        "f1_score": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }

