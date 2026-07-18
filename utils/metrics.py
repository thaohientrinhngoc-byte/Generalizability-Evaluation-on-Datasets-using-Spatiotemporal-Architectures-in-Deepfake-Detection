"""
Evaluation Metrics & Dataset Generalizability Score (DGS) Calculation.

Implements Section 3.3 Equation 2 from the paper:
    DGS = A_cross_mean * (A_cross_mean / A_internal) * (1 - sigma_cross)
"""

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    roc_curve,
    f1_score,
    classification_report,
)


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Computes Equal Error Rate (EER) using Brent's root-finding method.
    """
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    try:
        eer = float(brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0))
        return eer
    except Exception:
        return float("nan")


def compute_dgs_score(a_internal: float, a_cross_list: list) -> float:
    """
    Computes Dataset Generalizability Score (DGS) as introduced in Section 3.3 Equation 2:

        DGS = A_cross_bar * (A_cross_bar / A_internal) * (1 - sigma_cross)

    Args:
        a_internal (float): In-distribution test performance (e.g. AUC or Accuracy).
        a_cross_list (list of float): Out-of-distribution (cross-dataset) test performances.

    Returns:
        float: DGS score.
    """
    if not a_cross_list or a_internal <= 0:
        return 0.0

    a_cross_bar = float(np.mean(a_cross_list))
    sigma_cross = float(np.std(a_cross_list, ddof=0))  # standard deviation

    dgs = a_cross_bar * (a_cross_bar / a_internal) * (1.0 - sigma_cross)
    return float(dgs)


def evaluate_predictions(y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Evaluates ground truth vs predicted probabilities across all paper metrics.

    Args:
        y_true (np.ndarray): Binary ground truth array (0 for Real, 1 for Fake).
        y_scores (np.ndarray): Predicted probabilities (range [0, 1]).
        threshold (float): Classification decision threshold (default: 0.5).

    Returns:
        dict: Evaluation metrics dictionary.
    """
    y_pred_labels = (y_scores >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred_labels))
    try:
        auc = float(roc_auc_score(y_true, y_scores))
    except Exception:
        auc = float("nan")

    eer = calculate_eer(y_true, y_scores)
    f1 = float(f1_score(y_true, y_pred_labels, zero_division=0))

    report = classification_report(y_true, y_pred_labels, output_dict=True, zero_division=0)
    real_key = "0" if "0" in report else "0.0"
    fake_key = "1" if "1" in report else "1.0"

    real_prec = float(report.get(real_key, {}).get("precision", 0.0))
    fake_prec = float(report.get(fake_key, {}).get("precision", 0.0))
    real_rec = float(report.get(real_key, {}).get("recall", 0.0))
    fake_rec = float(report.get(fake_key, {}).get("recall", 0.0))

    return {
        "AUC": auc,
        "ACC": acc,
        "EER": eer,
        "F1": f1,
        "Recall_Real": real_rec,
        "Recall_Fake": fake_rec,
        "Precision_Real": real_prec,
        "Precision_Fake": fake_prec,
    }
