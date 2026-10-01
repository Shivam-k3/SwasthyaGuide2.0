"""
SWASTHYAGUIDE 2.0 — Heart Disease Risk module
Stage 3: model validation, robustness, calibration, explainability support.

Rules enforced here
-------------------
1.  The 20% holdout produced in Stage 2 stays LOCKED. Cross-validation,
    hyper-parameter search, threshold selection and calibration are all performed
    on the 80% training portion only. The holdout is touched exactly once, at the
    very end, by `final_holdout_evaluation()`.
2.  Every preprocessing step stays INSIDE the sklearn `Pipeline`. Imputers,
    encoders and the scaler are re-fitted from scratch inside every CV fold, so
    fold-test data can never influence them.
3.  Cohort membership is used for CV stratification and per-cohort reporting
    only. It is never a model feature.
4.  Nothing here selects or tunes a "medical decision threshold". Thresholds are
    reported as a research parameter and stay configurable.

Scope: research prototype for health RISK ASSESSMENT. Model outputs are research
risk scores requiring external validation. Not a medical device and not a
diagnostic system.
"""

from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from src import preprocessing as prep
from src.evaluate import DEFAULT_THRESHOLD, compute_metrics, confusion_matrix_frame_safe
from src.train import make_cohort_stratified_split

# --------------------------------------------------------------------------- #
# Stage 3 constants
# --------------------------------------------------------------------------- #

STAGE = "3 - validation, robustness, calibration, explainability"

#: CV design. Identical for all three models, so model differences are not an
#: artefact of a different resampling scheme.
CV_N_SPLITS = 5
CV_N_REPEATS = 5
CV_RANDOM_STATE = prep.RANDOM_SEED

#: Threshold grid. Reported, never auto-selected.
THRESHOLD_GRID = [0.30, 0.40, 0.50, 0.60, 0.70]

#: `ca` ablation: the pre-screening scope drops this feature entirely.
ABLATION_REMOVED_FEATURE = "ca"

#: Bounded search budget. Small on purpose — this stage is about validating
#: baselines, not about squeezing the last decimal from a noisy 920-row dataset.
SEARCH_N_ITER = 20
SEARCH_N_SPLITS = 5
SEARCH_RANDOM_STATE = prep.RANDOM_SEED

#: Metrics averaged over folds. Averaging several metrics is deliberate: ranking
#: on one metric would hide the trade-off between them.
CV_METRIC_NAMES = ["roc_auc", "pr_auc", "recall", "precision", "f1"]

#: Cohort reporting reliability thresholds.
MIN_COHORT_ROWS = 50
MIN_COHORT_CLASS_COUNT = 10

EXPLAINABILITY_DIR = prep.MODELS_DIR / "explainability"


# --------------------------------------------------------------------------- #
# Split handling
# --------------------------------------------------------------------------- #

def build_locked_split(df: pd.DataFrame, test_size: float = 0.2,
                       random_state: int = prep.RANDOM_SEED):
    """Reproduce the Stage 2 cohort-stratified split exactly.

    Same function, same seed, same parameters as Stage 2, so the holdout is the
    identical set of rows. That is what makes Stage 2's reported numbers
    comparable to Stage 3's.
    """
    train_idx, test_idx = make_cohort_stratified_split(df, test_size, random_state)
    return np.asarray(train_idx), np.asarray(test_idx)


def composite_strata(cohorts: pd.Series, target: pd.Series) -> pd.Series:
    """Cohort x label key used to stratify resampling.

    Plain stratification on the label alone would let cohort composition drift
    between folds. Disease prevalence ranges from 36% to 94% across cohorts, so
    the joint key keeps every fold compositionally comparable.
    """
    return cohorts.astype(str) + "__" + target.astype(str)


def make_cv_splits(strata: pd.Series, n_splits: int = CV_N_SPLITS,
                   n_repeats: int = CV_N_REPEATS,
                   random_state: int = CV_RANDOM_STATE) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pre-computed cross-validation fold indices.

    Computed once and reused by every model so all three see byte-identical
    folds. Validates up front that every stratum is large enough to appear in
    each fold; a stratum smaller than `n_splits` would silently break
    stratification rather than raise.
    """
    if strata.value_counts().min() < n_splits:
        small = strata.value_counts().loc[lambda s: s < n_splits]
        raise ValueError(
            "These cohort x label strata have fewer rows than n_splits, so they "
            f"cannot be stratified into every fold:\n{small.to_string()}"
        )

    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
    )
    dummy = np.zeros((len(strata), 1))
    splits = [
        (np.asarray(tr), np.asarray(va))
        for tr, va in splitter.split(dummy, strata.to_numpy())
    ]
    return splits


# --------------------------------------------------------------------------- #
# Feature-set variants
# --------------------------------------------------------------------------- #

def feature_set(drop: tuple[str, ...] = ()) -> tuple[list[str], list[str], list[str]]:
    """Return (feature_columns, numeric_branch, categorical_branch) for a variant.

    `drop` removes features from all three lists so the branch boundaries stay
    consistent. Used for the `ca` ablation.
    """
    features = [c for c in prep.FEATURE_COLUMNS if c not in drop]
    numeric = [c for c in (list(prep.NUMERIC_FEATURES) + list(prep.BOOLEAN_SOURCE_COLS))
               if c in features]
    categorical = [c for c in prep.CATEGORICAL_FEATURES if c in features]
    return features, numeric, categorical


def build_pipeline_for(model_name: str, feature_columns: list[str],
                       numeric_branch: list[str], categorical_branch: list[str],
                       params: dict | None = None,
                       random_state: int = prep.RANDOM_SEED):
    """Assemble a pipeline for one feature-set variant.

    Preprocessing is rebuilt for the variant rather than reused, so the ablation
    compares like with like: identical model family and identical procedure,
    only the feature availability differs.
    """
    pre = prep.build_preprocessor(
        numeric_features=numeric_branch, categorical_features=categorical_branch
    )
    base = prep.BASELINE_MODELS[model_name](random_state=random_state)
    if params:
        base.set_params(**params)
    from sklearn.pipeline import Pipeline
    return Pipeline([("preprocess", pre), ("model", base)])


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #

def run_cross_validation(pipeline, X: pd.DataFrame, y: pd.Series,
                         splits: list[tuple[np.ndarray, np.ndarray]],
                         threshold: float = DEFAULT_THRESHOLD) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit `pipeline` afresh on every fold, collecting per-fold metrics and OOF probabilities.

    A fresh `clone` per fold is what keeps preprocessing leakage-free: the
    imputers and encoders inside the pipeline are re-fitted on that fold's
    training rows alone.

    Returns (per_fold_metrics, oof_probabilities, oof_row_positions).
    """
    y_array = y.to_numpy()
    n = len(X)
    # With repeats a row is validated once per repeat. Accumulate the sum and
    # the count so the pooled prediction is the mean over repeats rather than
    # whichever repeat happened to be fitted last.
    oof_sum = np.zeros(n, dtype=float)
    oof_count = np.zeros(n, dtype=int)
    records = []

    for fold_no, (train_pos, val_pos) in enumerate(splits):
        fold_pipeline = clone(pipeline)
        fold_pipeline.fit(X.iloc[train_pos], y_array[train_pos])

        prob = fold_pipeline.predict_proba(X.iloc[val_pos])[:, 1]
        oof_sum[val_pos] += prob
        oof_count[val_pos] += 1

        metrics = compute_metrics(y_array[val_pos], prob, threshold)
        metrics["fold"] = fold_no
        records.append(metrics)

    if not np.all(oof_count > 0):
        missing = int((oof_count == 0).sum())
        raise RuntimeError(f"{missing} rows never received an out-of-fold prediction.")

    oof_prob = oof_sum / oof_count
    return pd.DataFrame.from_records(records), oof_prob, oof_count


def summarise_cv(per_fold: pd.DataFrame, model_name: str,
                 feature_variant: str = "full") -> dict:
    """Mean and standard deviation across folds for each metric."""
    out = {"model": model_name, "feature_variant": feature_variant,
           "n_folds": len(per_fold)}
    for metric in CV_METRIC_NAMES:
        values = per_fold[metric].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        out[f"{metric}_mean"] = float(values.mean()) if values.size else float("nan")
        out[f"{metric}_sd"] = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return out


def cohort_cv_metrics(y: pd.Series, oof_prob: np.ndarray, cohorts: pd.Series,
                      threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Pooled out-of-fold metrics per cohort, with a reliability flag.

    Each row's out-of-fold score is the mean of its per-repeat predictions, and
    every one of those predictions comes from a model that never saw that row.
    Pooled is used rather than averaged-per-fold because per-fold cohort slices
    are far too small to mean anything.
    """
    frame = pd.DataFrame({
        "target": y.to_numpy(),
        "prob": oof_prob,
        "cohort": cohorts.to_numpy(),
    })
    records = []
    for cohort, group in frame.groupby("cohort", sort=True):
        record = compute_metrics(group["target"].to_numpy(), group["prob"].to_numpy(), threshold)
        n_pos = int(group["target"].sum())
        n_neg = int(len(group) - n_pos)
        record["cohort"] = str(cohort)
        record["reliable"] = bool(
            len(group) >= MIN_COHORT_ROWS
            and n_pos >= MIN_COHORT_CLASS_COUNT
            and n_neg >= MIN_COHORT_CLASS_COUNT
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Hyper-parameter search
# --------------------------------------------------------------------------- #

#: Bounded search spaces. Each is deliberately narrow: a wide space on 736 rows
#: mostly fits noise, and a large search would also give the test set a chance to
#: leak in through repeated peeking.
PARAM_GRID: dict[str, dict] = {
    "logistic_regression": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        "model__class_weight": [None, "balanced"],
        "model__penalty": ["l2"],
    },
    "random_forest": {
        "model__n_estimators": [300, 500, 800],
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_leaf": [1, 2, 4, 8],
        "model__max_features": ["sqrt", 0.5, 1.0],
        "model__class_weight": [None, "balanced"],
    },
    "xgboost": {
        "model__n_estimators": [200, 400, 700],
        "model__max_depth": [2, 3, 4, 6],
        "model__learning_rate": [0.03, 0.06, 0.1, 0.2],
        "model__min_child_weight": [1, 3, 5],
        "model__subsample": [0.7, 0.9, 1.0],
        "model__reg_lambda": [1.0, 3.0, 10.0],
    },
}


def describe_search_space(model_name: str) -> dict:
    """Human-readable summary of the space, for documentation."""
    space = PARAM_GRID[model_name]
    total = int(np.prod([len(v) for v in space.values()]))
    n_iter = min(SEARCH_N_ITER, total)
    return {
        "model": model_name,
        "parameters": {k: [str(x) for x in v] for k, v in space.items()},
        "total_combinations": total,
        "n_iter_sampled": n_iter,
        "sampling": (
            "exhaustive grid (n_iter covers every combination)"
            if n_iter >= total
            else "random sampling WITH replacement; some combinations may be "
                 "drawn more than once and others not at all"
        ),
        "search_type": "RandomizedSearchCV",
        "cv_folds_used": SEARCH_N_SPLITS,
    }


def run_hyperparameter_search(pipeline, X: pd.DataFrame, y: pd.Series,
                              splits: list[tuple[np.ndarray, np.ndarray]],
                              model_name: str) -> dict:
    """Bounded randomised search using the SAME CV strategy as the baselines.

    Refit is on ROC-AUC, but every candidate is scored on ROC-AUC, PR-AUC, F1,
    recall and precision at once, so the selection decision can be checked
    against more than one number. Scoring happens on training-fold predictions
    only; the holdout is never passed to this function.
    """
    from sklearn.model_selection import RandomizedSearchCV

    cv_subset = splits[:SEARCH_N_SPLITS]  # first repeat only, to keep the budget bounded
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=PARAM_GRID[model_name],
        n_iter=min(SEARCH_N_ITER, int(np.prod([len(v) for v in PARAM_GRID[model_name].values()]))),
        scoring={
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
            "f1": "f1",
            "recall": "recall",
            "precision": "precision",
        },
        refit="roc_auc",
        cv=cv_subset,
        random_state=SEARCH_RANDOM_STATE,
        n_jobs=-1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X, y.to_numpy())

    results = pd.DataFrame(search.cv_results_)
    candidates = results[
        ["mean_test_roc_auc", "std_test_roc_auc", "mean_test_pr_auc", "mean_test_f1",
         "mean_test_recall", "mean_test_precision", "rank_test_roc_auc"]
    ].copy()
    candidates.index = [
        " | ".join(f"{k.split('__')[-1]}={v}" for k, v in p.items())
        for p in results["params"]
    ]
    space = describe_search_space(model_name)

    return {
        "model": model_name,
        "search": space,
        "best_params": {k.split("__")[-1]: v for k, v in search.best_params_.items()},
        "best_cv_roc_auc": float(search.best_score_),
        "best_cv_roc_auc_sd": float(
            results.loc[results["rank_test_roc_auc"].argmin(), "std_test_roc_auc"]
        ),
        "refit_pipeline": search.best_estimator_,
        "candidate_table": candidates.sort_values("rank_test_roc_auc").head(15),
        "n_candidates_scored": len(results),
        # With sampling-with-replacement and n_iter < total combinations, some of
        # the space is never evaluated. Recorded so the gap is explicit.
        "n_unique_combinations_scored": int(candidates.index.nunique()),
        "space_exhaustively_covered": bool(candidates.index.nunique() >= space["total_combinations"]),
    }


# --------------------------------------------------------------------------- #
# Threshold analysis
# --------------------------------------------------------------------------- #

def threshold_analysis(y: pd.Series, prob: np.ndarray, model_name: str,
                       thresholds: list[float] = THRESHOLD_GRID,
                       feature_variant: str = "full") -> pd.DataFrame:
    """Sensitivity, specificity, precision and F1 across a fixed threshold grid.

    Threshold choice is a research parameter. It is reported, never selected here,
    and stays configurable via `DEFAULT_THRESHOLD` / the CLI. Nothing in this
    table constitutes a recommended clinical cut-off.
    """
    y_array = y.to_numpy()
    records = []
    for threshold in thresholds:
        predicted = (prob >= threshold).astype(int)
        metrics = compute_metrics(y_array, prob, threshold)
        cm = confusion_matrix_frame_safe(y_array, prob, threshold)
        tn, fp, fn, tp = cm.iloc[0, 0], cm.iloc[0, 1], cm.iloc[1, 0], cm.iloc[1, 1]
        records.append({
            "model": model_name,
            "feature_variant": feature_variant,
            "threshold": threshold,
            "sensitivity_recall": metrics["recall"],
            "specificity": float(tn) / (float(tn) + float(fp)) if (tn + fp) else float("nan"),
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    return pd.DataFrame.from_records(records)


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #

def calibration_metrics(y: pd.Series, prob: np.ndarray, model_name: str,
                        n_bins: int = 10, feature_variant: str = "full") -> dict:
    """Brier score, expected calibration error, slope/intercept and bin-level data.

    The model output is a research risk score. A Brier score here describes how
    close the scores are to observed frequencies *in this dataset*; it is not a
    statement of clinically validated probability.
    """
    y_array = y.to_numpy().astype(int)
    prob = np.asarray(prob, dtype=float)

    brier = float(brier_score_loss(y_array, prob))
    prevalence = float(y_array.mean())

    # Bin-level reliability table (equal-width bins; population counts are kept).
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, n_bins - 1)
    bin_records = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        if count == 0:
            bin_records.append({
                "model": model_name, "bin": b, "n": 0,
                "mean_predicted": float("nan"), "observed_frequency": float("nan"),
                "gap": float("nan"),
            })
            continue
        mean_pred = float(prob[mask].mean())
        observed = float(y_array[mask].mean())
        ece += (count / len(y_array)) * abs(mean_pred - observed)
        bin_records.append({
            "model": model_name, "bin": b, "n": count,
            "mean_predicted": mean_pred, "observed_frequency": observed,
            "gap": mean_pred - observed,
        })

    # Calibration slope/intercept from a logistic recalibration fit.
    slope, intercept = _calibration_slope_intercept(y_array, prob)

    return {
        "model": model_name,
        "feature_variant": feature_variant,
        "n": int(len(y_array)),
        "prevalence": prevalence,
        "mean_predicted_probability": float(prob.mean()),
        "brier_score": brier,
        "brier_baseline_constant_prevalence": float(
            brier_score_loss(y_array, np.full(len(y_array), prevalence))
        ),
        "expected_calibration_error": float(ece),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "bins": pd.DataFrame.from_records(bin_records),
    }


def _calibration_slope_intercept(y_array: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    """Fit y ~ a + b*logit(p); b=1 and a=0 means perfect calibration.

    sklearn>=1.7 LogisticRegression `fit_intercept` handling differs, so this is
    fitted explicitly rather than relying on a helper.
    """
    from sklearn.linear_model import LogisticRegression

    epsilon = 1e-6
    clipped = np.clip(prob, epsilon, 1 - epsilon)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)

    with_slope = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    with_slope.fit(logits, y_array)
    slope = float(with_slope.coef_[0][0])
    intercept = float(with_slope.intercept_[0])

    intercept_only = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    intercept_only.fit(np.zeros((len(y_array), 1)), y_array)
    return slope, intercept


def plot_calibration_curves(bins_by_model: dict[str, pd.DataFrame], output_path: Path) -> Path:
    """Reliability diagram: observed frequency against mean predicted risk score.

    The diagonal is perfect agreement. Scores are research outputs, so the axis
    labels avoid presenting them as clinically validated probabilities; the
    figure only describes agreement inside this dataset.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    for model_name, bins in bins_by_model.items():
        usable = bins[bins["n"] > 0]
        ax.plot(usable["mean_predicted"], usable["observed_frequency"],
                marker="o", linewidth=1.6, label=model_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.2,
            label="perfect agreement")
    ax.set_xlabel("mean predicted risk score")
    ax.set_ylabel("observed outcome frequency")
    ax.set_title("Calibration of out-of-fold risk scores (selected models)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #

ERROR_GROUPS = {
    "dataset": lambda frame: frame["cohort"],
    "sex": lambda frame: frame["sex"],
    "missing_ca": lambda frame: np.where(frame["ca_missing"], "ca missing", "ca recorded"),
    "missing_thal": lambda frame: np.where(frame["thal_missing"], "thal missing", "thal recorded"),
    "missing_any": lambda frame: np.where(frame["any_missing"], "any missing", "complete"),
}

#: Feature groups used to check whether errors concentrate anywhere obvious.
MAJOR_FEATURE_GROUPS = {
    "oldpeak_gt_1": lambda frame: frame["oldpeak_clean"].gt(1.0),
    "exang_true": lambda frame: frame["exang_clean"].gt(0.5),
    "cp_asymptomatic": lambda frame: frame["cp"].eq("asymptomatic"),
    "male": lambda frame: frame["sex"].eq("Male"),
}


def build_error_frame(X: pd.DataFrame, y: pd.Series, prob: np.ndarray,
                      cohorts: pd.Series, threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """One row per scored observation, with a label for where it went wrong.

    `*_clean` columns hold the sentinel-masked values (chol==0, trestbps==0,
    oldpeak<0 -> NaN) so group membership reflects what the model actually saw.
    """
    masker = prep.InvalidValueMasker(columns=list(prep.NUMERIC_FEATURES) + list(prep.BOOLEAN_SOURCE_COLS))
    numeric_cols = list(prep.NUMERIC_FEATURES) + list(prep.BOOLEAN_SOURCE_COLS)
    cleaned = pd.DataFrame(
        masker.transform(X[numeric_cols].to_numpy(dtype=float)), columns=numeric_cols, index=X.index
    )

    frame = pd.DataFrame({
        "target": y.to_numpy(),
        "prob": np.asarray(prob, dtype=float),
        "cohort": cohorts.to_numpy(),
        "sex": X["sex"].to_numpy(),
        "cp": X["cp"].to_numpy(),
        "age": X["age"].to_numpy(),
    })
    frame["predicted"] = (frame["prob"] >= threshold).astype(int)
    frame["outcome"] = np.where(
        frame["target"] == frame["predicted"],
        np.where(frame["target"] == 1, "TP", "TN"),
        np.where(frame["target"] == 1, "FN", "FP"),
    )
    frame["is_error"] = frame["outcome"].isin(["FP", "FN"])
    frame["ca_missing"] = X["ca"].isna().to_numpy()
    frame["thal_missing"] = X["thal"].isna().to_numpy()
    frame["any_missing"] = X.isna().any(axis=1).to_numpy()
    for column in ("oldpeak_clean", "exang_clean", "chol_clean", "trestbps_clean", "ca_clean"):
        frame[column] = cleaned[column.replace("_clean", "")].to_numpy()
    return frame


def error_rates_by_group(error_frame: pd.DataFrame, grouping) -> pd.DataFrame:
    """Error rate within each level of a grouping."""
    keys = grouping(error_frame)
    records = []
    for level, group in error_frame.groupby(keys, sort=True):
        outcome = group["outcome"].value_counts()
        records.append({
            "level": str(level),
            "n": int(len(group)),
            "error_rate": float(group["is_error"].mean()),
            "fp_rate": float((group["outcome"] == "FP").mean()),
            "fn_rate": float((group["outcome"] == "FN").mean()),
            "prevalence": float(group["target"].mean()),
            "tn": int(outcome.get("TN", 0)), "fp": int(outcome.get("FP", 0)),
            "fn": int(outcome.get("FN", 0)), "tp": int(outcome.get("TP", 0)),
        })
    return pd.DataFrame.from_records(records)


def error_analysis_report(probabilities: dict[str, np.ndarray], X: pd.DataFrame,
                          y: pd.Series, cohorts: pd.Series,
                          threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Confusion counts and group-conditioned error rates for every model."""
    report: dict = {"threshold": threshold, "models": {}, "groups": {}}

    for model_name, prob in probabilities.items():
        frame = build_error_frame(X, y, prob, cohorts, threshold)
        counts = frame["outcome"].value_counts()
        report["models"][model_name] = {
            "tn": int(counts.get("TN", 0)), "fp": int(counts.get("FP", 0)),
            "fn": int(counts.get("FN", 0)), "tp": int(counts.get("TP", 0)),
            "error_rate": float(frame["is_error"].mean()),
            "fn_share_of_errors": float(
                counts.get("FN", 0) / max(int(frame["is_error"].sum()), 1)
            ),
        }
        # Stored as records, not DataFrames: a DataFrame would be stringified by
        # `json.dumps(default=str)` and become unreadable in the JSON card.
        for group_name, grouper in ERROR_GROUPS.items():
            report["groups"].setdefault(model_name, {})[group_name] = error_rates_by_group(
                frame, grouper
            ).to_dict(orient="records")
        for group_name, grouper in MAJOR_FEATURE_GROUPS.items():
            frame_with_flag = frame.assign(_flag=grouper(frame))
            report["groups"].setdefault(model_name, {})[group_name] = error_rates_by_group(
                frame_with_flag, lambda f: f["_flag"].map({True: "yes", False: "no"})
            ).to_dict(orient="records")
        # Continuous views.
        report["models"][model_name]["mean_prob_by_outcome"] = (
            frame.groupby("outcome")["prob"].mean().round(4).to_dict()
        )
        report["models"][model_name]["age_mean_by_outcome"] = (
            frame.groupby("outcome")["age"].mean().round(2).to_dict()
        )

    return report


# --------------------------------------------------------------------------- #
# Final holdout evaluation — called once, at the very end
# --------------------------------------------------------------------------- #

def final_holdout_evaluation(fitted_pipelines: dict[str, object],
                             X_test: pd.DataFrame, y_test: np.ndarray,
                             cohorts_test: pd.Series,
                             threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Single, locked evaluation of the finished pipelines on the untouched holdout.

    Called only after cross-validation, tuning, threshold and calibration work is
    complete. Its purpose is a clean estimate of development decisions that were
    made without ever seeing this data.
    """
    results = {}
    for model_name, pipeline in fitted_pipelines.items():
        prob = pipeline.predict_proba(X_test)[:, 1]
        results[model_name] = {
            "overall": compute_metrics(y_test, prob, threshold),
            "confusion_matrix": confusion_matrix_frame_safe(y_test, prob, threshold).to_dict(),
            "by_cohort": cohort_cv_metrics(
                pd.Series(y_test, index=X_test.index), prob, cohorts_test, threshold
            ).to_dict(orient="records"),
        }
    return results


# --------------------------------------------------------------------------- #
# Reporting helpers
# --------------------------------------------------------------------------- #

def environment_card() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "scikit_learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "shap": __import__("shap").__version__,
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
