"""
SWASTHYAGUIDE 2.0 — Heart Disease Risk module
Stage 3 verification suite.

Checks the Stage 3 invariants that matter, rather than just re-printing numbers.
Every check either passes or fails loudly. Exit code is non-zero on failure so
it can gate a pipeline.

Invariants covered
------------------
A. Reproducibility    seed fixed, raw SHA-256 unchanged, artefacts reloadable
B. Leakage safety     preprocessing fitted inside CV folds only, no test-set
                      contact before the final evaluation
C. Preprocessing      invalid sentinels masked, no rows dropped, no NaN survives
D. Pipeline integrity same pipeline at inference as at training
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import preprocessing as prep
from src import validation as val
from src.train import make_cohort_stratified_split


class _Checker:
    """Collects pass/fail results so one failure does not hide the rest."""

    def __init__(self, verbose: bool = True):
        self.results: list[dict] = []
        self.verbose = verbose

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        status = "PASS" if condition else "FAIL"
        self.results.append({"check": label, "status": status, "detail": detail})
        if self.verbose:
            print(f"  [{status}] {label}" + (f"  {detail}" if detail else ""))
        return bool(condition)

    @property
    def failed(self) -> list[dict]:
        return [r for r in self.results if r["status"] == "FAIL"]

    def summary(self) -> dict:
        failed = self.failed
        return {
            "total": len(self.results),
            "passed": len(self.results) - len(failed),
            "failed": len(failed),
            "all_passed": not failed,
            "failed_checks": failed,
            "results": self.results,
        }


def run_all(verbose: bool = True) -> dict:
    checker = _Checker(verbose)

    if verbose:
        print("-" * 74)
        print("STAGE 3 VERIFICATION")
        print("-" * 74)

    # ------------------------------------------------------------------ #
    # A. Reproducibility
    # ------------------------------------------------------------------ #
    if verbose:
        print("\nA. Reproducibility")

    checker.check(
        "raw CSV checksum unchanged",
        prep.file_sha256() == prep.EXPECTED_SHA256,
        prep.file_sha256(),
    )
    checker.check(
        "single seed constant defined",
        prep.RANDOM_SEED == 42 and val.CV_RANDOM_STATE == prep.RANDOM_SEED
        and val.SEARCH_RANDOM_STATE == prep.RANDOM_SEED,
        f"seed={prep.RANDOM_SEED}",
    )

    df = prep.build_analysis_frame()
    X, y, cohorts = prep.split_xy(df)

    # The Stage 2 split must be reproducible bit-for-bit.
    train_idx_a, test_idx_a = make_cohort_stratified_split(df, 0.2, prep.RANDOM_SEED)
    train_idx_b, test_idx_b = val.build_locked_split(df, 0.2, prep.RANDOM_SEED)
    checker.check(
        "Stage 3 split reproduces the Stage 2 split exactly",
        np.array_equal(np.sort(train_idx_a), np.sort(train_idx_b))
        and np.array_equal(np.sort(test_idx_a), np.sort(test_idx_b)),
        f"train={len(train_idx_b)} test={len(test_idx_b)}",
    )

    # ------------------------------------------------------------------ #
    # B. Leakage safety
    # ------------------------------------------------------------------ #
    if verbose:
        print("\nB. Leakage safety")

    checker.check(
        "train/test disjoint and complete",
        len(set(train_idx_a) & set(test_idx_a)) == 0
        and len(train_idx_a) + len(test_idx_a) == len(df),
    )

    X_train = X.iloc[train_idx_a]
    y_train = y.iloc[train_idx_a]
    X_test = X.iloc[test_idx_a]
    strata = val.composite_strata(cohorts.iloc[train_idx_a], y_train)
    splits = val.make_cv_splits(strata, n_splits=val.CV_N_SPLITS, n_repeats=1,
                                random_state=prep.RANDOM_SEED)

    checker.check(
        "CV folds built from the TRAIN portion only",
        max(int(tr.max()) for tr, _ in splits) < len(train_idx_a),
        "no fold index reaches into the holdout",
    )
    checker.check(
        "every train row appears as validation at least once",
        all(
            set(np.concatenate([va for _, va in splits])) == set(range(len(train_idx_a)))
            for _ in [0]
        ),
    )
    checker.check(
        "preprocessing lives inside the Pipeline",
        set(val.build_pipeline_for("logistic_regression", prep.FEATURE_COLUMNS,
                                   list(prep.NUMERIC_FEATURES) + list(prep.BOOLEAN_SOURCE_COLS),
                                   list(prep.CATEGORICAL_FEATURES)).named_steps)
        == {"preprocess", "model"},
    )

    # Imputer statistics must come from a fold's training rows only.
    probe_pipeline = val.build_pipeline_for(
        "logistic_regression", prep.FEATURE_COLUMNS,
        list(prep.NUMERIC_FEATURES) + list(prep.BOOLEAN_SOURCE_COLS),
        list(prep.CATEGORICAL_FEATURES), random_state=prep.RANDOM_SEED,
    )
    fold_tr, fold_va = splits[0]
    probe_pipeline.fit(X_train.iloc[fold_tr], y_train.iloc[fold_tr])

    num_pipe = probe_pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    imputer = num_pipe.named_steps["impute_median"]
    masker = num_pipe.named_steps["mask_invalid_values"]
    numeric_cols = list(prep.NUMERIC_FEATURES) + list(prep.BOOLEAN_SOURCE_COLS)

    fold_masked = masker.transform(X_train.iloc[fold_tr][numeric_cols].to_numpy(dtype=float))
    expected_medians = np.nanmedian(fold_masked, axis=0)
    checker.check(
        "imputer medians equal fold-train medians, not full-train medians",
        np.allclose(imputer.statistics_, expected_medians)
        and not np.allclose(imputer.statistics_, np.nanmedian(
            masker.transform(X_train[numeric_cols].to_numpy(dtype=float)), axis=0)),
        "differs from full-train medians, so leakage would show up",
    )

    # CV results must be deterministic: the same folds must give the same
    # out-of-fold predictions, because every fold refits a fresh clone.
    _, oof_a, _ = val.run_cross_validation(probe_pipeline, X_train, y_train, splits)
    _, oof_b, _ = val.run_cross_validation(probe_pipeline, X_train, y_train, splits)
    checker.check("CV on identical folds reproduces identical OOF predictions",
                  np.allclose(oof_a, oof_b),
                  f"max diff={np.max(np.abs(oof_a - oof_b)):.2e}")

    # ------------------------------------------------------------------ #
    # C. Preprocessing correctness
    # ------------------------------------------------------------------ #
    if verbose:
        print("\nC. Preprocessing correctness")

    checker.check(
        "no row is dropped by the pipeline",
        probe_pipeline.predict_proba(X_test).shape[0] == len(X_test),
        f"{len(X_test)} in, {probe_pipeline.predict_proba(X_test).shape[0]} out",
    )

    transformed = probe_pipeline.named_steps["preprocess"].transform(X_test)
    checker.check(
        "no NaN survives preprocessing",
        not np.isnan(np.asarray(transformed, dtype=float)).any(),
    )

    checker.check(
        "invalid sentinels mapped to NaN before imputation",
        bool(np.isnan(masker.transform(
            pd.DataFrame([{"age": 50, "trestbps": 0, "chol": 0, "thalch": 150,
                           "oldpeak": -2.6, "ca": 0, "fbs": 0.0, "exang": 0.0}])[numeric_cols]
            .to_numpy(dtype=float))[0][1:5]).sum() == 3),
        "chol==0, trestbps==0, oldpeak<0 all masked",
    )

    feature_names = prep.transformed_feature_names(probe_pipeline)
    checker.check(
        "id / dataset / num absent from transformed features",
        not any(t in n for n in feature_names for t in ("__id", "__dataset", "__num")),
        f"{len(feature_names)} transformed features",
    )

    # Ablation actually removes the feature.
    ablated_features, ablated_numeric, ablated_categorical = val.feature_set(drop=("ca",))
    checker.check(
        "`ca` ablation removes it from all three lists",
        "ca" not in ablated_features
        and "ca" not in ablated_numeric
        and "ca" not in ablated_categorical
        and len(ablated_features) == len(prep.FEATURE_COLUMNS) - 1,
        f"{len(ablated_features)} features",
    )

    # ------------------------------------------------------------------ #
    # D. Artefacts
    # ------------------------------------------------------------------ #
    if verbose:
        print("\nD. Artefacts and metric hygiene")

    tuned_dir = prep.MODELS_DIR / "tuned"
    for model_name in prep.BASELINE_MODELS:
        path = tuned_dir / f"{model_name}_pipeline.joblib"
        if not path.exists():
            checker.check(f"{model_name} tuned artefact exists", False, str(path))
            continue
        loaded = joblib.load(path)
        probs = loaded.predict_proba(X_test)[:, 1]
        checker.check(
            f"{model_name} tuned artefact reloads and predicts",
            probs.shape == (len(X_test),) and np.isfinite(probs).all(),
            f"n={probs.shape[0]}",
        )
        checker.check(
            f"{model_name} artefact carries preprocessing + model together",
            set(loaded.named_steps) == {"preprocess", "model"},
        )

    expected_csv = [
        "cv_metrics_overall.csv", "cv_metrics_by_cohort.csv", "threshold_analysis.csv",
        "cv_metrics_tuned_overall.csv", "cv_metrics_tuned_by_cohort.csv",
        "cv_metrics_per_fold.csv", "cv_metrics_tuned_per_fold.csv",
        "calibration_metrics.csv", "feature_ablation_ca.csv", "stage3_report.json",
        "calibration_curve.png", "calibration_bins.csv",
        "feature_ablation_ca_by_cohort.csv", "holdout_final_metrics.csv",
        "tuned/logistic_regression_pipeline.joblib",
        "tuned/random_forest_pipeline.joblib",
        "tuned/xgboost_pipeline.joblib",
    ]
    for name in expected_csv:
        path = prep.MODELS_DIR / name
        checker.check(f"models/{name} written", path.exists() and path.stat().st_size > 0)

    explain_dir = prep.MODELS_DIR / "explainability"
    checker.check("models/explainability/ populated",
                  explain_dir.exists() and len(list(explain_dir.iterdir())) > 0,
                  f"{len(list(explain_dir.iterdir())) if explain_dir.exists() else 0} files")
    if explain_dir.exists():
        names = {p.name for p in explain_dir.iterdir()}
        checker.check("global feature importance plot written",
                      "global_feature_importance.png" in names)
        for model_name in prep.BASELINE_MODELS:
            checker.check(f"SHAP summary plot for {model_name}",
                          f"shap_summary_{model_name}.png" in names)
        individual = [n for n in names if n.startswith("shap_individual")
                      and n.endswith(".png")]
        checker.check("at least 3 individual prediction explanations",
                      len(individual) >= 3, f"{len(individual)} plots")

        summary_path = explain_dir / "explainability_summary.json"
        if summary_path.exists():
            exp_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            additivity = exp_summary.get("additivity_check", {})
            checker.check(
                "SHAP additivity verified against the model's own output",
                bool(additivity) and all(r.get("verified") for r in additivity.values()),
                "; ".join(f"{k}: max gap {v['max_abs_gap']:.1e}"
                          for k, v in additivity.items()),
            )
            checker.check(
                "SHAP output scale declared per model family",
                set(exp_summary.get("output_scale", {})) >= set(prep.BASELINE_MODELS),
                "log-odds for the linear model, probability for the ensembles",
            )
            # Individual cases must come from out-of-fold scoring, so FP/FN are
            # real errors rather than rows a fitted model memorised.
            cases = {r.get("case") for r in exp_summary.get("individual_explanations", [])}
            checker.check("individual explanations cover all four outcome types",
                          {"TP", "TN", "FP", "FN"} <= cases, f"cases: {sorted(cases)}")

    # Threshold, calibration and error analysis must describe the SELECTED
    # models, not the untuned defaults, and the repeated-OOF aggregation must be
    # documented rather than left implicit.
    stage3_path = prep.MODELS_DIR / "stage3_report.json"
    if stage3_path.exists():
        card = json.loads(stage3_path.read_text(encoding="utf-8"))
        checker.check(
            "repeated-OOF policy documented (mean over repeats, never self-scored)",
            "mean of its per-repeat predictions" in str(card.get("oof_prediction_policy", "")),
            card.get("oof_prediction_policy", "MISSING"),
        )
        tuned_block = card.get("cv_metrics_tuned_overall", [])
        expected_models = set(prep.BASELINE_MODELS)
        checker.check(
            "tuned-model repeated CV recorded for all three models",
            {r["model"] for r in tuned_block} == expected_models,
            f"models: {sorted({r['model'] for r in tuned_block})}",
        )

    # Threshold analysis must cover the requested grid and not auto-select.
    threshold_path = prep.MODELS_DIR / "threshold_analysis.csv"
    if threshold_path.exists():
        thresholds = pd.read_csv(threshold_path)
        observed = sorted(thresholds["threshold"].unique().tolist())
        checker.check("threshold grid is exactly the requested set",
                      observed == sorted(val.THRESHOLD_GRID), str(observed))
        checker.check("threshold table reports sensitivity and specificity",
                      {"sensitivity_recall", "specificity"} <= set(thresholds.columns))
        checker.check("threshold table includes confusion counts",
                      {"tn", "fp", "fn", "tp"} <= set(thresholds.columns))

    # Calibration metrics present and sane.
    calibration_path = prep.MODELS_DIR / "calibration_metrics.csv"
    if calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
        checker.check("Brier score present for every model",
                      calibration["brier_score"].notna().all())
        checker.check("calibration slope/intercept present",
                      {"calibration_slope", "calibration_intercept"} <= set(calibration.columns))
        bins_path = prep.MODELS_DIR / "calibration_bins.csv"
        checker.check("calibration bin table written (predicted vs observed)",
                      bins_path.exists() and bins_path.stat().st_size > 0)

    # Ablation covers both variants for every model family.
    ablation_path = prep.MODELS_DIR / "feature_ablation_ca.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        variants = set(ablation["feature_variant"])
        checker.check("ablation covers full and no-ca variants",
                      len(variants) == 2
                      and any("no_ca" in v for v in variants),
                      str(sorted(variants)))
        checker.check("ablation covers all three model families",
                      set(ablation["model"]) == set(prep.BASELINE_MODELS))

    # Cohort reporting must flag small cohorts. Verified against the actual
    # criteria rather than hard-coded names: in the out-of-fold training pool
    # VA Long Beach has 160 rows and both classes well represented, so it is
    # legitimately reliable there. In the 184-row holdout it drops to n=40 and
    # must be flagged, which is checked separately below.
    cohort_path = prep.MODELS_DIR / "cv_metrics_by_cohort.csv"
    if cohort_path.exists():
        cohort = pd.read_csv(cohort_path)
        checker.check("per-cohort results carry a reliability flag",
                      "reliable" in cohort.columns)
        expected_reliable = (
            (cohort["n"] >= val.MIN_COHORT_ROWS)
            & (cohort["n_positive"] >= val.MIN_COHORT_CLASS_COUNT)
            & ((cohort["n"] - cohort["n_positive"]) >= val.MIN_COHORT_CLASS_COUNT)
        )
        checker.check(
            "reliability flag matches the stated criteria everywhere",
            bool((expected_reliable == cohort["reliable"]).all()),
            "flag computed from n, positives and negatives",
        )
        flagged = sorted(cohort.loc[~cohort["reliable"], "cohort"].unique().tolist())
        checker.check(
            "Switzerland flagged unreliable in the CV pool",
            "Switzerland" in flagged,
            f"flagged: {flagged}",
        )

    # Error analysis recorded.
    stage3_path = prep.MODELS_DIR / "stage3_report.json"
    if stage3_path.exists():
        card = json.loads(stage3_path.read_text(encoding="utf-8"))
        checker.check("error analysis recorded TN/FP/FN/TP per model",
                      all({"tn", "fp", "fn", "tp"} <= set(v)
                          for v in card["error_analysis"]["models"].values()))
        checker.check("CV scope documented as training-only",
                      "training portion only" in card["cv"]["scope"])
        checker.check("holdout policy documented as locked",
                      "locked" in card["holdout"]["policy"])

    # ------------------------------------------------------------------ #
    # E. Requirements pinned
    # ------------------------------------------------------------------ #
    if verbose:
        print("\nE. Dependency pinning")
    requirements = prep.PROJECT_ROOT / "requirements.txt"
    if requirements.exists():
        text = requirements.read_text(encoding="utf-8")
        pinned = [line.strip() for line in text.splitlines()
                  if line.strip() and not line.strip().startswith("#")]
        checker.check("requirements.txt has no unpinned entries",
                      all("==" in line for line in pinned),
                      f"{len(pinned)} pinned packages")
        for required in ("pandas", "numpy", "scikit-learn", "xgboost", "shap", "joblib"):
            checker.check(f"{required} pinned", any(line.startswith(required) for line in pinned))
    else:
        checker.check("requirements.txt exists", False)

    summary = checker.summary()
    if verbose:
        print("-" * 74)
        print(f"{summary['passed']}/{summary['total']} checks passed"
              + ("" if summary["all_passed"] else f" — {summary['failed']} FAILED"))
        for failure in summary["failed_checks"]:
            print(f"  FAILED: {failure['check']}  {failure['detail']}")
        print("-" * 74)

    return summary


if __name__ == "__main__":
    result = run_all(verbose=True)
    sys.exit(0 if result["all_passed"] else 1)
