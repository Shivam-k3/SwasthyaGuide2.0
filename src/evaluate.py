"""
SWASTHYAGUIDE 2.0 — Heart Disease Risk module
Stage 2: evaluation metrics and cohort-aware reporting.

Metric policy
-------------
ROC-AUC and PR-AUC are threshold-free and are the primary metrics. Because the
positive class is the majority class (55.3%), accuracy is reported but treated as
secondary — it is dominated by the base rate and is not evidence of skill.
Recall, precision and F1 are reported at a fixed 0.5 decision threshold and are
therefore threshold-dependent; the threshold is stated alongside them rather
than tuned on the test set.

Per-cohort results carry a `reliable` flag. A cohort with few positives or few
rows produces a wide, meaningless confidence interval, so those figures are
labelled unreliable instead of being presented as findings.

Scope: research prototype for health RISK ASSESSMENT. Not a diagnostic system.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_THRESHOLD = 0.5

#: Below this many rows, or this few positives/negatives, a cohort's metrics are
#: reported but explicitly flagged as not statistically meaningful.
MIN_COHORT_ROWS = 50
MIN_COHORT_CLASS_COUNT = 10


def compute_metrics(
    y_true,
    y_prob,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Full metric bundle for one (y_true, y_prob) pair.

    `roc_auc` and `pr_auc` use the probabilities; the rest use the hard
    prediction at `threshold`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    out: dict = {
        "n": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "prevalence": float(y_true.mean()) if y_true.size else float("nan"),
        "threshold": float(threshold),
    }

    # Threshold-free metrics are undefined when only one class is present.
    if np.unique(y_true).size == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")

    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def confusion_frame(y_true, y_prob, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Confusion matrix as a labelled 2x2 frame (rows = actual, cols = predicted)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    index = ["actual: no disease (0)", "actual: disease (1)"]
    columns = ["predicted: no disease (0)", "predicted: disease (1)"]
    return pd.DataFrame(cm, index=index, columns=columns)


def confusion_matrix_frame_safe(
    y_true, y_prob, threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """Confusion matrix as raw counts, without row/column labels.

    Used where the counts are needed programmatically (specificity, error
    analysis) rather than for display. `confusion_frame` is the labelled version
    for printing.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    return pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=[0, 1]),
        columns=["pred_negative", "pred_positive"],
    )


def specificity(y_true, y_prob, threshold: float = DEFAULT_THRESHOLD) -> float:
    """True-negative rate, i.e. TN / (TN + FP). Not returned by sklearn directly."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob, dtype=float) >= threshold).astype(int)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    return float(tn) / float(denominator) if denominator else float("nan")


def per_cohort_metrics(
    y_true,
    y_prob,
    cohorts: pd.Series,
    threshold: float = DEFAULT_THRESHOLD,
) -> pd.DataFrame:
    """Metrics broken down by source cohort, with a statistical-reliability flag."""
    frame = pd.DataFrame(
        {
            "target": np.asarray(y_true).astype(int),
            "prob": np.asarray(y_prob, dtype=float),
            "cohort": pd.Series(cohorts).reset_index(drop=True),
        }
    )

    records = []
    for name, group in frame.groupby("cohort", sort=True):
        record = compute_metrics(group["target"].to_numpy(), group["prob"].to_numpy(), threshold)
        n_pos = int(group["target"].sum())
        n_neg = int(len(group) - n_pos)
        record["cohort"] = str(name)
        record["reliable"] = bool(
            len(group) >= MIN_COHORT_ROWS
            and n_pos >= MIN_COHORT_CLASS_COUNT
            and n_neg >= MIN_COHORT_CLASS_COUNT
        )
        records.append(record)

    result = pd.DataFrame.from_records(records)
    columns = [
        "cohort",
        "n",
        "n_positive",
        "prevalence",
        "roc_auc",
        "pr_auc",
        "recall",
        "precision",
        "f1",
        "accuracy",
        "tn",
        "fp",
        "fn",
        "tp",
        "reliable",
    ]
    return result[[c for c in columns if c in result.columns]]


def format_metric_table(metrics_by_model: dict[str, dict]) -> pd.DataFrame:
    """Model x metric comparison table."""
    rows = []
    for name, m in metrics_by_model.items():
        rows.append(
            {
                "model": name,
                "roc_auc": m.get("roc_auc"),
                "pr_auc": m.get("pr_auc"),
                "recall": m.get("recall"),
                "precision": m.get("precision"),
                "f1": m.get("f1"),
                "accuracy": m.get("accuracy"),
                "n": m.get("n"),
                "n_positive": m.get("n_positive"),
                "threshold": m.get("threshold"),
            }
        )
    table = pd.DataFrame.from_records(rows).set_index("model")
    return table.round(4)


def format_cohort_table(cohort_metrics_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Cohort x model comparison table with the reliability flag carried through."""
    frames = []
    for name, frame in cohort_metrics_by_model.items():
        part = frame.copy()
        part["model"] = name
        frames.append(part)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.set_index(["model", "cohort"])
    return combined.round(4)
