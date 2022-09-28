from typing import Dict

import numpy as np
import sklearn.metrics as skm


def get_fpr_tpr_thr(y_true, y_pred, pos_label):
    """Computes the FPR, TPR and THRESHOLD for a binary classification problem.

        * `y_score >= threhold` is classified as `pos_label`.

    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
        pos_label ([type]): [description]

    Returns:
        fpr : Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= `thresholds[i]`.
        tpr : Increasing true positive rates such that element `i` is the true
            positive rate of predictions with score >= `thresholds[i]`.
        thresholds : Decreasing thresholds on the decision function used to compute
            fpr and tpr. `thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`.
    """
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred, pos_label=pos_label)
    return fpr, tpr, thresholds


def get_precision_recall_thr(y_true, y_pred, pos_label=1):
    precision, recall, thresholds = skm.precision_recall_curve(
        y_true, y_pred, pos_label=pos_label
    )
    return precision, recall, thresholds


def compute_aupr(precision, recall):
    return skm.auc(recall, precision)


def compute_fpr_tpr_thr_given_tpr_level(fpr, tpr, thresholds, tpr_level):
    if all(tpr < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    idxs = [i for i, x in enumerate(tpr) if x >= tpr_level]
    idx = min(idxs)
    return fpr[idx], tpr[idx], thresholds[idx]


def compute_detection_error(op_fpr, op_tpr, pos_ratio):
    """Return the misclassification probability when TPR is fixed."""
    # Get ratios of positives to negatives
    neg_ratio = 1 - pos_ratio
    # Get indexes of all TPR >= fixed tpr level
    detection_error = pos_ratio * (1 - op_tpr) + neg_ratio * op_fpr
    # Return the minimum detection error such that TPR >= 0.95
    return detection_error


def compute_detection_metrics(
    fixed_tpr_scores: np.ndarray, detect_scores: np.ndarray, tpr_level=0.95, pos_label=1
) -> Dict[str, float]:
    """Compute OOD detection metrics. If `score >= threshold`, the sample is classified
    as `in-distribution`.

    * `score >= threhold` is classified as `pos_label` (correct -> tpr).

    Args:
        fixed_tpr_scores (np.ndarray): This is the in-distribution data's scores for.
        detect_scores (np.ndarray): This is the Out-of-Distribution data's scores
        tpr_level (float, optional): Fixed TPR. Defaults to 0.95.

    Returns (dict) with keys:
        tnr_at_{tpr_level}_tpr,
        fpr_at_{tpr_level}_tpr,
        detection_error,
        auroc,
        aupr_in,
        aupr_out,
        thr
    """
    # configured to detect in-distribution samples if pos_label=1
    pos = pos_label * np.ones(len(fixed_tpr_scores))
    neg = (1 - pos_label) * np.ones(len(detect_scores))

    y_true = np.concatenate([pos, neg])
    y_pred = np.concatenate([fixed_tpr_scores.reshape(-1), detect_scores.reshape(-1)])
    pos_ratio = sum(np.array(y_true) == pos_label) / len(y_true)

    fpr, tpr, thresholds = get_fpr_tpr_thr(y_true, y_pred, pos_label)
    fpr_at_given_tpr, best_tpr, thr = compute_fpr_tpr_thr_given_tpr_level(
        fpr, tpr, thresholds, tpr_level
    )
    detection_error = compute_detection_error(fpr_at_given_tpr, best_tpr, pos_ratio)
    auroc = skm.auc(fpr, tpr)

    precision, recall, _ = get_precision_recall_thr(y_true, y_pred, pos_label)
    aupr_in = compute_aupr(precision, recall)

    precision, recall, _ = get_precision_recall_thr(1 - y_true, -y_pred, pos_label)
    aupr_out = compute_aupr(precision, recall)

    return {
        f"tnr_at_{tpr_level}_tpr": 1 - float(fpr_at_given_tpr),
        f"fpr_at_{tpr_level}_tpr": float(fpr_at_given_tpr),
        "detection_error": float(detection_error),
        "auroc": float(auroc),
        "aupr_in": float(aupr_in),
        "aupr_out": float(aupr_out),
        "thr": float(thr),
    }


METRICS_NAMES_PRETTY = {
    "tnr_at_0.95_tpr": "TNR at 95% TPR",
    "fpr_at_0.95_tpr": "FPR at 95% TPR",
    "detection_error": "Detection error",
    "auroc": "AUROC",
    "aupr_in": "AUPR in",
    "aupr_out": "AUPR out",
}
