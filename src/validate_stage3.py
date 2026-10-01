"""
SWASTHYAGUIDE 2.0 — Heart Disease Risk module
Stage 3 driver: repeated cross-validation, bounded tuning, threshold analysis,
calibration, the `ca` ablation, error analysis and explainability artefacts.

Execution order, and why it is this order
-----------------------------------------
1. Rebuild the Stage 2 split so the holdout is the identical set of rows.
2. Everything development-related happens on the 80% training portion:
   cross-validation, hyper-parameter search, thresholds, calibration, ablation.
3. Only once all decisions are frozen does `final_holdout_evaluation` touch the
   20% holdout, exactly once.

That ordering is the whole point. Repeatedly scoring the holdout while making
choices quietly turns it into a validation set and inflates the final number.

Usage:
    python -m src.validate_stage3
    python -m src.validate_stage3 --n-iter 20 --cv-repeats 5
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import preprocessing as prep
from src import validation as val

MODEL_ORDER = ["logistic_regression", "random_forest", "xgboost"]


def _paired_comparison(per_fold_frames: list[pd.DataFrame], label: str,
                       metric: str = "roc_auc") -> pd.DataFrame:
    """Paired per-fold differences between every model pair.

    All models are evaluated on identical folds, so the paired difference is the
    correct uncertainty estimate. Comparing one model's SD against another's
    mean overstates the uncertainty: fold difficulty is shared and largely
    cancels in the difference.

    The sign is reported as a two-sided consistency count rather than as a p
    value: with 25 folds, the 12-sample binomial reference is coarse, and a
    significance test here would invite exactly the "this model is better"
    reading this project is not entitled to claim. A difference is only called
    out when the mean gap is at least as large as the paired SD and the
    consistent side wins clearly more than half the folds.
    """
    _say("")
    _say(f"Paired per-fold differences in {metric} ({label}, same 25 folds):")
    # The frames arrive with a leading "model" column, so rebuild the per-model
    # series explicitly rather than relying on a positional zip.
    columns = {}
    for frame in per_fold_frames:
        name = str(frame["model"].iloc[0])
        columns[name] = frame.set_index("fold")[metric]
    wide = pd.DataFrame(columns)
    n_folds = len(wide)

    rows = []
    for i, left in enumerate(MODEL_ORDER):
        for right in MODEL_ORDER[i + 1:]:
            diff = (wide[left] - wide[right]).dropna()
            mean_diff = float(diff.mean())
            sd_diff = float(diff.std(ddof=1)) if len(diff) > 1 else float("nan")
            wins = int((diff > 0).sum())
            # "consistent" = the gap is larger than its own paired spread AND
            # the same side wins at least 80% of folds.
            consistent = bool(abs(mean_diff) >= sd_diff
                              and max(wins, len(diff) - wins) / len(diff) >= 0.8)
            rows.append({
                "comparison": f"{left} - {right}",
                "mean_diff": mean_diff,
                "sd_diff": sd_diff,
                f"folds_first_better": wins,
                f"of_{n_folds}": len(diff),
                "gap_exceeds_paired_sd": consistent,
            })
    table = pd.DataFrame(rows)
    _say(table.round(4).to_string(index=False))
    separable = table[table["gap_exceeds_paired_sd"]]
    if separable.empty:
        _say("Read: no pair is separated by these folds. The three models are")
        _say("statistically indistinguishable on ROC-AUC here.")
    else:
        for _, row in separable.iterrows():
            _say(f"Read: {row['comparison']} ({row['mean_diff']:+.4f}) is separated - "
                 f"the gap exceeds its own paired spread.")
    _say("")
    _say("Separable on this metric is NOT clinical superiority, and does not")
    _say("transfer to a new population.")
    return table


def _say(*args):
    print(*args)


def run(n_iter: int = val.SEARCH_N_ITER, cv_repeats: int = val.CV_N_REPEATS,
        threshold: float = 0.5, test_size: float = 0.2,
        random_state: int = prep.RANDOM_SEED) -> dict:
    """Full Stage 3 run. Returns a summary dict and writes everything to models/."""
    np.random.seed(random_state)
    val.SEARCH_N_ITER = n_iter
    val.CV_N_REPEATS = cv_repeats

    _say("=" * 78)
    _say("SWASTHYAGUIDE 2.0 - Heart Disease Risk | STAGE 3")
    _say("=" * 78)
    _say(f"Dataset            : {prep.DATA_PATH}")
    _say(f"SHA-256 (verified) : {prep.file_sha256()}")
    _say(f"Seed               : {random_state}")
    _say(f"CV design          : {val.CV_N_SPLITS}-fold x {cv_repeats} repeats, "
         f"stratified on cohort x label")
    _say(f"Search budget      : {n_iter} candidates per model, "
         f"{val.SEARCH_N_SPLITS}-fold")
    _say("Scope              : research prototype, risk ASSESSMENT. Not diagnostic.")
    _say("")

    # ------------------------------------------------------------------ #
    # 1. Data and the locked holdout
    # ------------------------------------------------------------------ #
    df = prep.build_analysis_frame()
    X, y, cohorts = prep.split_xy(df)
    train_idx, test_idx = val.build_locked_split(df, test_size, random_state)

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx].to_numpy()
    cohorts_train, cohorts_test = cohorts.iloc[train_idx], cohorts.iloc[test_idx]

    _say("-" * 78)
    _say("LOCKED SPLIT (identical to Stage 2: same function, same seed)")
    _say("-" * 78)
    _say(f"train {len(X_train)} rows (prevalence {y_train.mean():.4f})")
    _say(f"test  {len(X_test)} rows (prevalence {y_test.mean():.4f})  <-- HELD BACK")
    _say("")

    strata_train = val.composite_strata(cohorts_train, y.iloc[train_idx])
    cv_splits = val.make_cv_splits(strata_train, val.CV_N_SPLITS, cv_repeats, random_state)
    _say(f"CV folds built on TRAIN ONLY: {len(cv_splits)} folds")
    _say("")

    features, numeric_branch, categorical_branch = val.feature_set()

    # ------------------------------------------------------------------ #
    # 2. Repeated cross-validation of the baselines
    # ------------------------------------------------------------------ #
    _say("-" * 78)
    _say("1. REPEATED STRATIFIED CROSS-VALIDATION (un-tuned baselines)")
    _say("-" * 78)

    cv_records: list[dict] = []
    cv_by_cohort: list[pd.DataFrame] = []
    oof_probabilities: dict[str, np.ndarray] = {}
    oof_per_fold: dict[str, pd.DataFrame] = {}

    for model_name in MODEL_ORDER:
        pipeline = val.build_pipeline_for(
            model_name, features, numeric_branch, categorical_branch, random_state=random_state
        )
        per_fold, oof_prob, _ = val.run_cross_validation(pipeline, X_train, y_train, cv_splits, threshold)
        oof_per_fold[model_name] = per_fold
        oof_probabilities[model_name] = oof_prob

        cv_records.append(val.summarise_cv(per_fold, model_name, "full"))
        cohort_frame = val.cohort_cv_metrics(y_train, oof_prob, cohorts_train, threshold)
        cohort_frame.insert(0, "model", model_name)
        cv_by_cohort.append(cohort_frame)

        summary = cv_records[-1]
        _say(f"\n{model_name}:")
        _say(f"  ROC-AUC  {summary['roc_auc_mean']:.4f} +/- {summary['roc_auc_sd']:.4f}")
        _say(f"  PR-AUC   {summary['pr_auc_mean']:.4f} +/- {summary['pr_auc_sd']:.4f}")
        _say(f"  Recall   {summary['recall_mean']:.4f} +/- {summary['recall_sd']:.4f}")
        _say(f"  Prec.    {summary['precision_mean']:.4f} +/- {summary['precision_sd']:.4f}")
        _say(f"  F1       {summary['f1_mean']:.4f} +/- {summary['f1_sd']:.4f}")
    _say("")

    cv_overall = pd.DataFrame(cv_records)
    cv_overall.to_csv(prep.MODELS_DIR / "cv_metrics_overall.csv", index=False)
    cv_by_cohort_df = pd.concat(cv_by_cohort, ignore_index=True)
    cv_by_cohort_df.to_csv(prep.MODELS_DIR / "cv_metrics_by_cohort.csv", index=False)

    # Per-fold rows are kept so any model-vs-model comparison can be made
    # PAIRED. Comparing two summary means against each other's SD is the wrong
    # test: all models share the same folds, so the paired difference has a much
    # smaller SD than either model alone.
    per_fold_all = [
        oof_per_fold[model_name].assign(model=model_name) for model_name in MODEL_ORDER
    ]
    pd.concat(per_fold_all, ignore_index=True).to_csv(
        prep.MODELS_DIR / "cv_metrics_per_fold.csv", index=False
    )

    _paired_comparison(per_fold_all, "un-tuned baselines")

    _say(cv_overall.round(4).to_string(index=False))
    _say("")
    _say("Mean +/- SD across folds. Comparing one model's SD against another's")
    _say("mean is NOT a valid test here: all models share the same 25 folds, so")
    _say("fold difficulty is common to both sides. The paired comparison below")
    _say("is the honest one.")
    _say("")

    # ------------------------------------------------------------------ #
    # 3. Bounded hyper-parameter search, same CV strategy
    # ------------------------------------------------------------------ #
    _say("-" * 78)
    _say("2. BOUNDED HYPER-PARAMETER SEARCH (training folds only)")
    _say("-" * 78)

    search_results = {}
    tuned_pipelines = {}
    candidate_tables = []

    for model_name in MODEL_ORDER:
        pipeline = val.build_pipeline_for(
            model_name, features, numeric_branch, categorical_branch, random_state=random_state
        )
        result = val.run_hyperparameter_search(pipeline, X_train, y_train, cv_splits, model_name)
        search_results[model_name] = result
        tuned_pipelines[model_name] = result["refit_pipeline"]

        space = result["search"]
        _say(f"\n{model_name}:")
        _say(f"  space          : {space['parameters']}")
        _say(f"  combinations   : {space['total_combinations']} possible, "
             f"{space['n_iter_sampled']} sampled, "
             f"{result['n_unique_combinations_scored']} unique")
        _say(f"  coverage       : {'exhaustive' if result['space_exhaustively_covered'] else 'PARTIAL - not every combination was scored'}")
        _say(f"  scoring        : ROC-AUC (refit), also PR-AUC / F1 / recall / precision")
        _say(f"  best CV ROC-AUC: {result['best_cv_roc_auc']:.4f} "
             f"(+/- {result['best_cv_roc_auc_sd']:.4f} across the {val.SEARCH_N_SPLITS} search folds)")
        _say(f"  best params    : {result['best_params']}")
        _say("  NOTE: the search used the first repeat only, to keep the budget bounded.")
        _say("  NOTE: better CV score is not evidence of clinical superiority.")
        candidate_tables.append(result["candidate_table"])

    _say("")

    # ------------------------------------------------------------------ #
    # 4. Repeated CV of the selected configurations
    # ------------------------------------------------------------------ #
    _say("-" * 78)
    _say("3. REPEATED CROSS-VALIDATION OF THE SELECTED CONFIGURATIONS")
    _say("-" * 78)
    _say("Same folds, same procedure as the baselines above, but with each model's")
    _say("selected hyper-parameters. All later development analysis (thresholds,")
    _say("calibration, error analysis) uses these out-of-fold predictions, so it")
    _say("describes the selected models rather than the untuned defaults.")
    _say("")

    tuned_cv_records: list[dict] = []
    tuned_cv_by_cohort: list[pd.DataFrame] = []
    tuned_oof_probabilities: dict[str, np.ndarray] = {}
    tuned_oof_per_fold: dict[str, pd.DataFrame] = {}
    tuned_per_fold_all: list[pd.DataFrame] = []

    for model_name in MODEL_ORDER:
        pipeline = val.build_pipeline_for(
            model_name, features, numeric_branch, categorical_branch,
            params=search_results[model_name]["best_params"], random_state=random_state
        )
        per_fold, oof_prob, _ = val.run_cross_validation(
            pipeline, X_train, y_train, cv_splits, threshold
        )
        tuned_oof_per_fold[model_name] = per_fold
        tuned_oof_probabilities[model_name] = oof_prob
        tuned_cv_records.append(val.summarise_cv(per_fold, model_name, "full"))
        tuned_per_fold_all.append(per_fold.assign(model=model_name))

        cohort_frame = val.cohort_cv_metrics(y_train, oof_prob, cohorts_train, threshold)
        cohort_frame.insert(0, "model", model_name)
        tuned_cv_by_cohort.append(cohort_frame)

        baseline = next(r for r in cv_records if r["model"] == model_name)
        selected = tuned_cv_records[-1]
        _say(f"\n{model_name}:")
        for metric in val.CV_METRIC_NAMES:
            delta = selected[f"{metric}_mean"] - baseline[f"{metric}_mean"]
            _say(f"  {metric:<8} {selected[f'{metric}_mean']:.4f} +/- "
                 f"{selected[f'{metric}_sd']:.4f}   (baseline {baseline[f'{metric}_mean']:.4f}, "
                 f"delta {delta:+.4f})")
    _say("")

    tuned_cv_overall = pd.DataFrame(tuned_cv_records)
    tuned_cv_overall.to_csv(prep.MODELS_DIR / "cv_metrics_tuned_overall.csv", index=False)
    tuned_cv_by_cohort_df = pd.concat(tuned_cv_by_cohort, ignore_index=True)
    tuned_cv_by_cohort_df.to_csv(prep.MODELS_DIR / "cv_metrics_tuned_by_cohort.csv", index=False)
    pd.concat(tuned_per_fold_all, ignore_index=True).to_csv(
        prep.MODELS_DIR / "cv_metrics_tuned_per_fold.csv", index=False
    )

    _paired_comparison(tuned_per_fold_all, "selected configurations")

    # Every downstream development analysis switches to the tuned OOF scores.
    oof_probabilities = tuned_oof_probabilities
    oof_per_fold = tuned_oof_per_fold

    # ------------------------------------------------------------------ #
    # 5. Threshold analysis on out-of-fold TRAINING predictions
    # ------------------------------------------------------------------ #
    _say("-" * 78)
    _say("4. THRESHOLD ANALYSIS (selected models, out-of-fold train predictions)")
    _say("-" * 78)
    _say(f"grid: {val.THRESHOLD_GRID}")
    _say("Threshold is a research parameter. No value is selected or recommended.")
    _say("")

    threshold_frames = []
    for model_name in MODEL_ORDER:
        frame = val.threshold_analysis(y_train, oof_probabilities[model_name], model_name)
        threshold_frames.append(frame)
        _say(f"\n{model_name}:")
        _say(frame[["threshold", "sensitivity_recall", "specificity", "precision",
                    "f1", "tn", "fp", "fn", "tp"]].round(4).to_string(index=False))
    threshold_df = pd.concat(threshold_frames, ignore_index=True)
    threshold_df.to_csv(prep.MODELS_DIR / "threshold_analysis.csv", index=False)
    _say("")

    # ------------------------------------------------------------------ #
    # 5. Calibration on out-of-fold TRAINING predictions
    # ------------------------------------------------------------------ #
    _say("-" * 78)
    _say("5. CALIBRATION (selected models, out-of-fold train predictions)")
    _say("-" * 78)
    _say("Model outputs are research risk scores. A score here is not a clinically")
    _say("validated probability and carries no diagnostic meaning.")
    _say("")

    calibration_records = []
    calibration_bins = []
    for model_name in MODEL_ORDER:
        result = val.calibration_metrics(y_train, oof_probabilities[model_name], model_name)
        calibration_bins.append(result.pop("bins"))
        calibration_records.append(result)
        _say(f"{model_name}:")
        _say(f"  Brier score                 : {result['brier_score']:.4f}")
        _say(f"  Brier (constant prevalence) : {result['brier_baseline_constant_prevalence']:.4f}")
        _say(f"  Expected calibration error   : {result['expected_calibration_error']:.4f}")
        _say(f"  Calibration slope (ideal 1)  : {result['calibration_slope']:.4f}")
        _say(f"  Calibration intercept (ideal 0): {result['calibration_intercept']:.4f}")
        _say(f"  Mean predicted probability   : {result['mean_predicted_probability']:.4f}")
        _say(f"  Observed prevalence          : {result['prevalence']:.4f}")
        _say("")

    calibration_df = pd.DataFrame(calibration_records)
    calibration_df.to_csv(prep.MODELS_DIR / "calibration_metrics.csv", index=False)
    all_bins = pd.concat(calibration_bins, ignore_index=True)
    all_bins.to_csv(prep.MODELS_DIR / "calibration_bins.csv", index=False)

    val.plot_calibration_curves(
        {name: frame for name, frame in zip(MODEL_ORDER, calibration_bins)},
        prep.MODELS_DIR / "calibration_curve.png",
    )
    _say(f"reliability diagram written: {(prep.MODELS_DIR / 'calibration_curve.png').name}")
    _say("")

    # ------------------------------------------------------------------ #
    # 6. `ca` ablation — feature-availability experiment
    # ------------------------------------------------------------------ #
    _say("-" * 78)
    _say("6. FEATURE ABLATION: `ca` REMOVED (pre-screening scope)")
    _say("-" * 78)
    _say("`ca` is an angiographic vessel-narrowing count and may be unavailable in an")
    _say("ordinary pre-screening interaction. This measures the cost of that gap.")
    _say("Same model families, same folds, same procedure - only availability differs.")
    _say("")

    ablation_features, ablation_numeric, ablation_categorical = val.feature_set(
        drop=(val.ABLATION_REMOVED_FEATURE,)
    )
    ablation_records = []
    ablation_cohort = []

    for model_name in MODEL_ORDER:
        tuned_params = search_results[model_name]["best_params"]
        for variant, (cols, num, cat) in {
            "full_13_features": (features, numeric_branch, categorical_branch),
            f"no_{val.ABLATION_REMOVED_FEATURE}_12_features": (ablation_features, ablation_numeric, ablation_categorical),
        }.items():
            pipeline = val.build_pipeline_for(
                model_name, cols, num, cat, params=tuned_params, random_state=random_state
            )
            per_fold, oof_prob, _ = val.run_cross_validation(pipeline, X_train, y_train, cv_splits, threshold)
            ablation_records.append(val.summarise_cv(per_fold, model_name, variant))
            cohort_frame = val.cohort_cv_metrics(y_train, oof_prob, cohorts_train, threshold)
            cohort_frame.insert(0, "feature_variant", variant)
            cohort_frame.insert(0, "model", model_name)
            ablation_cohort.append(cohort_frame)

    ablation_df = pd.DataFrame(ablation_records)
    ablation_df.to_csv(prep.MODELS_DIR / "feature_ablation_ca.csv", index=False)
    pd.concat(ablation_cohort, ignore_index=True).to_csv(
        prep.MODELS_DIR / "feature_ablation_ca_by_cohort.csv", index=False
    )

    _say(ablation_df[["model", "feature_variant", "n_folds"] + [f"{m}_mean" for m in val.CV_METRIC_NAMES]
                     + [f"{m}_sd" for m in val.CV_METRIC_NAMES]].round(4).to_string(index=False))
    _say("")
    _say("This is a feature-availability / deployment-scope measurement, not a")
    _say("claim that either variant is clinically preferable.")
    _say("")

    # ------------------------------------------------------------------ #
    # 7. Error analysis on out-of-fold TRAIN predictions
    # ------------------------------------------------------------------ #
    _say("-" * 78)
    _say("7. ERROR ANALYSIS (selected models, out-of-fold train predictions)")
    _say("-" * 78)

    error_report = val.error_analysis_report(oof_probabilities, X_train, y_train, cohorts_train, threshold)
    for model_name in MODEL_ORDER:
        counts = error_report["models"][model_name]
        _say(f"\n{model_name}: TN={counts['tn']} FP={counts['fp']} "
             f"FN={counts['fn']} TP={counts['tp']}  error rate={counts['error_rate']:.4f}")
        _say(f"  FN share of errors: {counts['fn_share_of_errors']:.4f}")
        _say(f"  mean probability by outcome: {counts['mean_prob_by_outcome']}")
        _say(f"  mean age by outcome: {counts['age_mean_by_outcome']}")
        _say("  error rate by cohort:")
        # Group tables live in the JSON card as records; rebuild a frame to print.
        _say(pd.DataFrame(error_report["groups"][model_name]["dataset"])
             .round(4).to_string(index=False))
        _say("  error rate by any-missing-vs-complete:")
        _say(textwrap.indent(
            pd.DataFrame(error_report["groups"][model_name]["missing_any"])
            .round(4).to_string(index=False), "    "))
    _say("")
    _say("These are ASSOCIATIONS within the data. They are not causal claims and")
    _say("carry no clinical interpretation.")
    _say("")

    # ------------------------------------------------------------------ #
    # 8. Explainability artefacts
    # ------------------------------------------------------------------ #
    _say("-" * 78)
    _say("8. EXPLAINABILITY ARTEFACTS")
    _say("-" * 78)

    from src import explainability as exp

    explainability_summary = exp.generate_all(
        fitted_pipelines=tuned_pipelines,
        X_explain=X_train,
        feature_columns=features,
        output_dir=val.EXPLAINABILITY_DIR,
        random_state=random_state,
        max_individual=5,
        oof_probabilities=oof_probabilities,
        y_explain=y_train,
    )
    for line in explainability_summary["console_lines"]:
        _say(line)
    _say("")

    # ------------------------------------------------------------------ #
    # 9. Persist tuned pipelines and the stage record
    # ------------------------------------------------------------------ #
    tuned_dir = prep.MODELS_DIR / "tuned"
    tuned_dir.mkdir(parents=True, exist_ok=True)
    for model_name, pipeline in tuned_pipelines.items():
        joblib.dump(pipeline, tuned_dir / f"{model_name}_pipeline.joblib")

    stage_card = {
        "module": "heart_disease_risk",
        "stage": val.STAGE,
        "generated_utc": val.utc_now(),
        "disclaimer": (
            "Research prototype for health risk assessment. Model outputs are research "
            "risk scores that require external validation. Not a medical device and not "
            "a diagnostic system."
        ),
        "data": {
            "path": "data/heart_disease_uci.csv",
            "sha256": prep.file_sha256(),
            "rows": int(len(df)),
        },
        "seed": random_state,
        "cv": {
            "n_splits": val.CV_N_SPLITS,
            "n_repeats": cv_repeats,
            "stratification": "cohort x target",
            "total_folds": len(cv_splits),
            "scope": "training portion only; holdout excluded",
        },
        "holdout": {
            "policy": "locked; used once after all development decisions were frozen",
            "test_size": test_size,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
        "features": {
            "full": features,
            "ablated": ablation_features,
            "excluded": prep.EXCLUDED_FROM_FEATURES,
            "transformed_full": prep.transformed_feature_names(
                val.build_pipeline_for("logistic_regression", features, numeric_branch,
                                       categorical_branch, random_state=random_state)
                .fit(X_train, y_train)
            ),
        },
        "cv_metrics_overall": cv_overall.to_dict(orient="records"),
        "cv_metrics_tuned_overall": tuned_cv_overall.to_dict(orient="records"),
        "oof_prediction_policy": (
            "each row's out-of-fold score is the mean of its per-repeat predictions; "
            "every repeat's model never saw that row"
        ),
        "search": {
            name: {
                "space": result["search"],
                "best_params": result["best_params"],
                "best_cv_roc_auc": result["best_cv_roc_auc"],
                "best_cv_roc_auc_sd": result["best_cv_roc_auc_sd"],
                # How much of the declared grid was actually evaluated. With
                # sampling-with-replacement and a bounded budget this can be far
                # short of the space, which must not read as an exhaustive grid.
                "n_candidates_scored": result["n_candidates_scored"],
                "n_unique_combinations_scored": result["n_unique_combinations_scored"],
                "space_exhaustively_covered": result["space_exhaustively_covered"],
            }
            for name, result in search_results.items()
        },
        "threshold_grid": val.THRESHOLD_GRID,
        "threshold_analysis": threshold_df.to_dict(orient="records"),
        "calibration": calibration_df.to_dict(orient="records"),
        "ablation": ablation_df.to_dict(orient="records"),
        "error_analysis": error_report,
        "explainability_files": sorted(p.name for p in val.EXPLAINABILITY_DIR.iterdir()),
        "environment": val.environment_card(),
    }

    # The single, final look at the holdout. Everything above was decided without it.
    _say("-" * 78)
    _say("9. FINAL HOLDOUT EVALUATION (first and only look)")
    _say("-" * 78)
    holdout_results = val.final_holdout_evaluation(
        tuned_pipelines, X_test, y_test, cohorts_test, threshold
    )
    for model_name, result in holdout_results.items():
        overall = result["overall"]
        _say(f"\n{model_name}: ROC-AUC={overall['roc_auc']:.4f} PR-AUC={overall['pr_auc']:.4f} "
             f"recall={overall['recall']:.4f} precision={overall['precision']:.4f} "
             f"F1={overall['f1']:.4f} accuracy={overall['accuracy']:.4f}")
    _say("")
    _say("CV means above are the development estimates. These holdout numbers are")
    _say("reported for completeness and have not informed any decision.")
    _say("")

    stage_card["holdout_final"] = holdout_results
    pd.DataFrame([
        {"model": name, **{k: v for k, v in res["overall"].items()}}
        for name, res in holdout_results.items()
    ]).to_csv(prep.MODELS_DIR / "holdout_final_metrics.csv", index=False)

    val.write_json(prep.MODELS_DIR / "stage3_report.json", stage_card)

    # ------------------------------------------------------------------ #
    # 10. Reproducibility verification
    # ------------------------------------------------------------------ #
    from src import verify_stage3

    _say("-" * 78)
    _say("10. VERIFICATION SUITE")
    _say("-" * 78)
    verification = verify_stage3.run_all(verbose=True)
    _say("")

    _say("=" * 78)
    _say("STAGE 3 COMPLETE — artefacts in models/")
    _say("=" * 78)
    for path in sorted(prep.MODELS_DIR.rglob("*")):
        if path.is_file():
            _say(f"  {path.relative_to(prep.PROJECT_ROOT)}")
    _say("=" * 78)
    _say("STOPPING HERE. No FastAPI, no Streamlit, no deployment.")

    return {
        "cv_overall": cv_overall,
        "cv_by_cohort": cv_by_cohort_df,
        "search": search_results,
        "threshold": threshold_df,
        "calibration": calibration_df,
        "ablation": ablation_df,
        "errors": error_report,
        "holdout": holdout_results,
        "verification": verification,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWASTHYAGUIDE 2.0 Stage 3 validation.")
    parser.add_argument("--n-iter", type=int, default=val.SEARCH_N_ITER,
                        help="Candidates per model for the bounded search.")
    parser.add_argument("--cv-repeats", type=int, default=val.CV_N_REPEATS,
                        help="Repeats for repeated cross-validation.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Reporting threshold. A research parameter, not tuned.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=prep.RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run(n_iter=args.n_iter, cv_repeats=args.cv_repeats, threshold=args.threshold,
        test_size=args.test_size, random_state=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
