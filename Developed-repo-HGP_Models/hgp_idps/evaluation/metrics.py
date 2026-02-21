"""
Evaluation Metrics  (Section VII, Appendix J-K)
===============================================

- Standard binary classification: accuracy, precision, recall, F1, AUC-ROC
- Calibration: ECE, Brier score, reliability diagram data
- Per-attack / per-CVE F1
- Adversarial degradation curve
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    auc,
    classification_report,
    confusion_matrix,
)
from typing import Dict, Optional


# ------------------------------------------------------------------ #
#  Core detection metrics                                              #
# ------------------------------------------------------------------ #
def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_scores: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute full metric suite.

    Returns dict with accuracy, precision, recall, f1, auc_roc, fpr.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # FPR
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    auc_val = 0.0
    if y_scores is not None and len(np.unique(y_true)) > 1:
        auc_val = roc_auc_score(y_true, y_scores)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc_val,
        "fpr": fpr,
    }


# ------------------------------------------------------------------ #
#  Expected Calibration Error                                          #
# ------------------------------------------------------------------ #
def expected_calibration_error(
    y_true: np.ndarray,
    confidences: np.ndarray,
    predictions: np.ndarray,
    n_bins: int = 15,
) -> Dict:
    """Compute ECE and reliability-diagram data.

    Parameters
    ----------
    y_true : array [N]
    confidences : array [N]  — model confidence ∈ [0, 1]
    predictions : array [N]  — binary predictions
    n_bins : int

    Returns
    -------
    dict with "ece", "brier" (if scores supplied), and "bins" list.
    """
    bins_data = []
    ece = 0.0
    boundaries = np.linspace(0, 1, n_bins + 1)

    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = (predictions[mask] == y_true[mask]).mean()
        bin_conf = confidences[mask].mean()
        weight = mask.mean()
        ece += weight * abs(bin_acc - bin_conf)
        bins_data.append({
            "confidence": float(bin_conf),
            "accuracy": float(bin_acc),
            "count": int(mask.sum()),
        })

    return {"ece": float(ece), "bins": bins_data}


# ------------------------------------------------------------------ #
#  Per-attack F1                                                       #
# ------------------------------------------------------------------ #
def per_attack_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    attack_labels: np.ndarray,
) -> Dict[str, float]:
    """Compute F1 per original attack category.

    Parameters
    ----------
    y_true : binary labels
    y_pred : binary predictions
    attack_labels : original multi-class attack names
    """
    results = {}
    for label in np.unique(attack_labels):
        mask = attack_labels == label
        if mask.sum() == 0:
            continue
        results[str(label)] = f1_score(y_true[mask], y_pred[mask], zero_division=0)
    return results
