"""Metrics and threshold selection for the RSNA CNN baseline."""
from __future__ import annotations

import numpy as np


def _check_labels(y) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.size == 0 or not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError(f'labels must be binary (0/1); got {sorted(set(np.unique(y)))[:5]}')
    return y


def _check_probs(p, n: int) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    if p.shape != (n,):
        raise ValueError(f'probability shape {p.shape} does not match {n} labels')
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('probabilities must be finite and within [0, 1]')
    return p


def compute_metrics(y_true, y_prob, threshold: float) -> dict:
    """AUROC plus threshold metrics and the confusion matrix.

    y_prob is P(class = 1); predictions are y_prob >= threshold.
    """
    y = _check_labels(y_true)
    p = _check_probs(y_prob, len(y))
    if len(np.unique(y)) < 2:
        raise ValueError('metrics require both classes present in y_true')
    from sklearn.metrics import roc_auc_score
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = sens
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        'auroc': float(roc_auc_score(y, p)),
        'threshold': float(threshold),
        'sensitivity': sens,
        'specificity': spec,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
                             'matrix': [[tn, fp], [fn, tp]]},
        'n': int(len(y)),
        'n_positive': int(y.sum()),
        'n_negative': int(len(y) - y.sum()),
    }


def select_threshold_youden(y_true, y_prob) -> float:
    """Threshold maximising sensitivity + specificity - 1.

    Must be computed on VALIDATION data only; the test set stays untouched
    until final evaluation. Ties keep the lowest threshold among the best.
    """
    y = _check_labels(y_true)
    p = _check_probs(y_prob, len(y))
    if len(np.unique(y)) < 2:
        raise ValueError('threshold selection requires both classes in y_true')
    best_t, best_j = 0.5, -np.inf
    for t in np.unique(p):                    # predict positive when p >= t
        pred = p >= t
        tp = int(((pred == 1) & (y == 1)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t
