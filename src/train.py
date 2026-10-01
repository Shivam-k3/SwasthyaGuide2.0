"""
SWASTHYAGUIDE 2.0 — Heart Disease Risk module
Stage 2: baseline training and evaluation.

What this script does
---------------------
1. Loads the raw CSV read-only (checksum verified).
2. Derives the binary target `num == 0 -> 0`, `num > 0 -> 1` in memory.
3. Creates a cohort-stratified train/test split so every cohort appears in both
   parts with a comparable class balance.
4. Fits three un-tuned baselines (Logistic Regression, Random Forest, XGBoost)
   on top of ONE shared preprocessing pipeline, so the comparison is fair.
5. Reports ROC-AUC, PR-AUC, recall, precision, F1, accuracy and a confusion
   matrix, overall and per cohort.
6. Saves the complete pipeline (preprocessing + model) per model with joblib,
   plus a JSON card recording the data hash, features, metrics and versions.

Split methodology
-----------------
`train_test_split(..., stratify=cohort x label)`. Plain stratification on the
label alone would not guarantee every cohort lands in both parts, and cohort
membership strongly determines disease prevalence (36.2% Hungary vs 93.5%
Switzerland). Stratifying on the joint label keeps each cohort present in train
and test with a similar class balance.

No hyper-parameter search is performed. No model is selected as "best" — the
full comparison is printed and the trade-offs are left visible.

Scope: research prototype for health RISK ASSESSMENT. Not a diagnostic system.

Usage:
    python -m src.train
    python -m src.train --test-size 0.2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

if __package__ in (None, ""):  # allow `python src/train.py`
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import preprocessing as prep
from src.evaluate import (
    DEFAULT_THRESHOLD,
    compute_metrics,
    confusion_frame,
    format_cohort_table,
    format_metric_table,
    per_cohort_metrics,
)

TARGET_COLUMN = prep.TARGET_COL
COHORT_COLUMN = prep.COHORT_COL
RAW_TARGET = prep.RAW_TARGET_COL


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def make_cohort_stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = prep.RANDOM_SEED,
):
    """Train/test split stratified on the cohort x label combination.

    Returns train_idx, test_idx as positional indices into `df`.
    """
    strata = df[COHORT_COLUMN].astype(str) + "__" + df[TARGET_COLUMN].astype(str)
    if strata.value_counts().min() < 2:
        raise ValueError(
            "A cohort x label group has fewer than 2 rows; stratification is impossible."
        )
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=strata,
    )
    return train_idx, test_idx


def describe_split(df: pd.DataFrame, train_idx, test_idx) -> pd.DataFrame:
    """Row counts and class balance per cohort for both partitions."""
    rows = []
    for cohort, group in df.groupby(COHORT_COLUMN, sort=True):
        n = len(group)
        in_train = int(np.isin(group.index.to_numpy(), train_idx).sum())
        in_test = int(np.isin(group.index.to_numpy(), test_idx).sum())
        rows.append(
            {
                "cohort": cohort,
                "total_rows": n,
                "train_rows": in_train,
                "test_rows": in_test,
                "overall_prevalence": round(float(group[TARGET_COLUMN].mean()), 4),
                "train_prevalence": round(
                    float(group.loc[np.isin(group.index.to_numpy(), train_idx), TARGET_COLUMN].mean()), 4
                ),
                "test_prevalence": round(
                    float(group.loc[np.isin(group.index.to_numpy(), test_idx), TARGET_COLUMN].mean()), 4
                ),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Main training routine
# --------------------------------------------------------------------------- #

def run_training(test_size: float = 0.2, random_state: int = prep.RANDOM_SEED,
                 threshold: float = DEFAULT_THRESHOLD, verbose: bool = True) -> dict:
    def say(*args):
        if verbose:
            print(*args)

    np.random.seed(random_state)

    # 1. Data — raw CSV stays untouched.
    df = prep.build_analysis_frame()
    X, y, cohorts = prep.split_xy(df)

    say("=" * 78)
    say("SWASTHYAGUIDE 2.0 - Heart Disease Risk | Stage 2 baseline training")
    say("=" * 78)
    say(f"Dataset            : {prep.DATA_PATH}")
    say(f"SHA-256 (verified) : {prep.file_sha256()}")
    say(f"Rows / features    : {len(df)} rows, {len(prep.FEATURE_COLUMNS)} raw features")
    say(f"Excluded           : {prep.EXCLUDED_FROM_FEATURES}")
    say(f"Label rule         : num == 0 -> 0 (no disease), num > 0 -> 1 (disease)")
    say(f"Class balance      : {dict(y.value_counts().sort_index())}")
    say(f"Positive rate      : {y.mean():.4f}")
    say("")

    # 2. Split.
    train_idx, test_idx = make_cohort_stratified_split(df, test_size, random_state)
    split_summary = describe_split(df, train_idx, test_idx)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx].to_numpy(), y.iloc[test_idx].to_numpy()
    cohorts_train = cohorts.iloc[train_idx]
    cohorts_test = cohorts.iloc[test_idx]

    say("-" * 78)
    say(f"Cohort-stratified split (test_size={test_size}, seed={random_state})")
    say("-" * 78)
    say(split_summary.to_string(index=False))
    say("")
    say(f"train rows: {len(X_train)} | test rows: {len(X_test)}")
    say(f"train prevalence: {y_train.mean():.4f} | test prevalence: {y_test.mean():.4f}")
    say("")

    # 3. Missingness BEFORE imputation, so the report reflects the raw input.
    say("-" * 78)
    say("Missing values per feature (train), before imputation")
    say("-" * 78)
    say(X_train.isna().sum().to_string())
    say("")

    # 4. Invalid sentinels actually present, for the record.
    invalid_counts = {}
    for col, rule in prep.INVALID_VALUE_RULES.items():
        col_values = X_train[col].to_numpy(dtype=float)
        if rule["op"] == "==":
            hits = int(np.sum(col_values == rule["value"]))
        else:
            hits = int(np.sum(col_values < rule["value"]))
        invalid_counts[col] = hits
    say("Invalid sentinel values found in train (mapped to NaN inside the pipeline):")
    for col, hits in invalid_counts.items():
        say(f"  {col:8s} {prep.INVALID_VALUE_RULES[col]['op']} "
            f"{prep.INVALID_VALUE_RULES[col]['value']:.0f}  ->  {hits} rows")
    say("")

    # 5. Fit the three baselines on one shared pipeline.
    metrics_by_model: dict[str, dict] = {}
    cohort_metrics_by_model: dict[str, pd.DataFrame] = {}
    confusion_by_model: dict[str, pd.DataFrame] = {}
    saved_paths: dict[str, str] = {}
    feature_names: dict[str, list[str]] = {}
    estimators: dict[str, object] = {}

    prep.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for name, factory in prep.BASELINE_MODELS.items():
        model = factory(random_state=random_state)
        pipeline = prep.build_pipeline(model)

        # Fitted on TRAIN ONLY. Imputers, encoders and the scaler therefore see
        # nothing from the test set, and the saved artefact carries that exact
        # preprocessing forward to inference.
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]

        estimators[name] = pipeline
        metrics_by_model[name] = compute_metrics(y_test, y_prob, threshold)
        cohort_metrics_by_model[name] = per_cohort_metrics(
            y_test, y_prob, cohorts_test.reset_index(drop=True), threshold
        )
        confusion_by_model[name] = confusion_frame(y_test, y_prob, threshold)
        feature_names[name] = prep.transformed_feature_names(pipeline)

        artefact = prep.MODELS_DIR / f"{name}_pipeline.joblib"
        joblib.dump(pipeline, artefact)
        saved_paths[name] = str(artefact.relative_to(prep.PROJECT_ROOT))

        say(f"fitted + saved: {name:22s} -> {saved_paths[name]}")

    say("")

    # 6. Overall comparison.
    metric_table = format_metric_table(metrics_by_model)
    say("-" * 78)
    say(f"OVERALL TEST-SET METRICS (decision threshold = {threshold})")
    say("-" * 78)
    say(metric_table.to_string())
    say("")
    say("Primary metrics are ROC-AUC and PR-AUC (threshold-free).")
    say("Accuracy is secondary: the positive class is the majority (55.3%), so a")
    say("constant 'always disease' predictor already scores 0.553 accuracy.")
    say("Recall / precision / F1 depend on the threshold and were NOT tuned.")
    say("")

    # 7. Confusion matrices.
    say("-" * 78)
    say("CONFUSION MATRICES (test set)")
    say("-" * 78)
    for name, frame in confusion_by_model.items():
        say(f"\n{name}:")
        say(frame.to_string())
    say("")

    # 8. Cohort-wise results.
    cohort_table = format_cohort_table(cohort_metrics_by_model)
    say("-" * 78)
    say("COHORT-WISE TEST-SET METRICS")
    say("-" * 78)
    say(cohort_table[["n", "n_positive", "prevalence", "roc_auc", "pr_auc",
                      "recall", "precision", "f1", "accuracy", "reliable"]].to_string())
    say("")
    say("'reliable' is False when a cohort has < 50 test rows or < 10 positives or")
    say("< 10 negatives in the test set. Those figures are shown for completeness")
    say("but are not statistically meaningful.")
    say("")

    # 9. Persist the numeric results alongside the pipelines.
    metric_table.to_csv(prep.MODELS_DIR / "baseline_metrics_overall.csv")
    cohort_table.to_csv(prep.MODELS_DIR / "baseline_metrics_by_cohort.csv")
    split_summary.to_csv(prep.MODELS_DIR / "split_summary.csv", index=False)

    card = {
        "module": "heart_disease_risk",
        "stage": "2 — preprocessing + un-tuned baselines",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "disclaimer": (
            "Research prototype for health risk assessment. Not a medical device, "
            "not a diagnostic system, not validated for clinical decision-making."
        ),
        "data": {
            "path": str(prep.DATA_PATH.relative_to(prep.PROJECT_ROOT)),
            "sha256": prep.file_sha256(),
            "rows": int(len(df)),
            "raw_features": prep.FEATURE_COLUMNS,
            "excluded_from_features": prep.EXCLUDED_FROM_FEATURES,
            "cohort_column_used_for": ["split stratification", "per-cohort reporting"],
        },
        "target": {
            "raw_column": RAW_TARGET,
            "derived_column": TARGET_COLUMN,
            "rule": "num == 0 -> 0 (no disease); num > 0 -> 1 (disease)",
            "positive_rate": round(float(y.mean()), 6),
        },
        "invalid_value_rules": {
            col: {"op": rule["op"], "value": rule["value"], "reason": rule["reason"],
                  "train_rows_affected": invalid_counts.get(col, 0)}
            for col, rule in prep.INVALID_VALUE_RULES.items()
        },
        "ca_decision": {
            "treatment": "ordinal numeric (median imputation + missingness indicator)",
            "justification": (
                "Values are integral counts {0,1,2,3}; disease prevalence is monotonic "
                "in the count (26.5 / 68.7 / 80.5 / 85.0 %); Spearman(ca, target) "
                "rho=0.485, p=1.4e-19. Median imputation writes 0 for the 66.4% of rows "
                "with missing ca, a healthy-looking value, so a ca_missing indicator is "
                "attached to correct that direction."
            ),
        },
        "split": {
            "strategy": "cohort-stratified train_test_split",
            "strata": "dataset x target",
            "test_size": test_size,
            "random_state": random_state,
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "models": {
            name: {
                "pipeline_file": saved_paths[name],
                "estimator": type(estimators[name].named_steps["model"]).__name__,
                "hyperparameters": {
                    k: v for k, v in estimators[name].named_steps["model"].get_params().items()
                    if not callable(v)
                },
                "tuned": False,
                "n_transformed_features": len(feature_names[name]),
                "transformed_feature_names": feature_names[name],
                "metrics_overall": metrics_by_model[name],
                "metrics_by_cohort": cohort_metrics_by_model[name].to_dict(orient="records"),
                "confusion_matrix": {
                    "labels": ["no disease (0)", "disease (1)"],
                    "values": confusion_by_model[name].to_numpy().tolist(),
                },
            }
            for name in prep.BASELINE_MODELS
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "Small dataset (920 rows); metrics have wide uncertainty.",
            "Cohort sample sizes are uneven, so most per-cohort test metrics are unreliable.",
            "No hyper-parameter tuning was performed; these are baselines, not tuned models.",
            "No external validation set; results are on a single held-out split.",
            "Risk-assessment research prototype only. Not a diagnostic system.",
        ],
    }
    (prep.MODELS_DIR / "baseline_report.json").write_text(
        json.dumps(card, indent=2, default=str), encoding="utf-8"
    )

    say("-" * 78)
    say("Artefacts written to models/:")
    for path in sorted(p for p in prep.MODELS_DIR.iterdir()):
        say(f"  {path.name:42s} {path.stat().st_size:>9,d} bytes")
    say("=" * 78)

    return {
        "metrics_overall": metric_table,
        "metrics_by_cohort": cohort_table,
        "confusion": confusion_by_model,
        "split_summary": split_summary,
        "card": card,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SWASTHYAGUIDE heart disease baselines.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction held out for testing (default 0.2).")
    parser.add_argument("--seed", type=int, default=prep.RANDOM_SEED,
                        help=f"Random seed (default {prep.RANDOM_SEED}).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Decision threshold (default {DEFAULT_THRESHOLD}, not tuned).")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_training(test_size=args.test_size, random_state=args.seed, threshold=args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
