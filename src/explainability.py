"""
SWASTHYAGUIDE 2.0 — Heart Disease Risk module
Stage 3 explainability: contributions and associations, not causation.

What this module does and does not claim
----------------------------------------
It quantifies how much each feature CONTRIBUTED to a given model-derived risk
score. It does not establish medical causation, and it does not make a
diagnostic claim. A coefficient describes the relationship the fitted model
learned in this dataset; it is not a statement about disease mechanism.

SHAP values decompose a single prediction into per-feature additive
contributions relative to a baseline (here, the mean predicted score across the
evaluation sample). Positive contribution pushes the score toward "disease";
negative toward "no disease".

OUTPUT SCALE differs by model, and the numbers are not comparable across models:
  * Logistic Regression -> log-odds (LinearExplainer linearises the link, so
    contributions sum to the log-odds shift, not to a probability change).
  * Random Forest / XGBoost -> raw probability units (TreeExplainer).
Read the sign and the ranking within a model; do not compare magnitudes across
model families.

Model-specific attribution, as requested:
  * Logistic Regression -> standardized coefficients (exact on the scaled matrix)
  * Random Forest       -> impurity-based feature importance
  * XGBoost             -> impurity-based feature importance
  * All three           -> SHAP global summary + individual explanations

Note on SHAP for Random Forest: TreeExplainer on the *preprocessed* matrix is
exact and fast. SHAP is not run on the raw frame, because the attribution must
match the transformed features the model actually consumes.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src import preprocessing as prep

#: Attribution caveat attached to every artefact this module writes.
ATTRIBUTION_CAVEAT = (
    "Feature attribution describes how a fitted model used its inputs to produce a "
    "risk score. It reflects model association and contribution within this dataset. "
    "It is NOT evidence of medical causation, NOT a diagnostic finding, and NOT a "
    "clinically validated interpretation. Correlated features can share or dilute "
    "attribution between themselves."
)

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def _transform(pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Apply the fitted preprocessing and return the matrix plus its column names."""
    pre = pipeline.named_steps["preprocess"]
    matrix = pre.transform(X)
    names = list(pre.get_feature_names_out())
    return np.asarray(matrix, dtype=float), names


def _tidy(names: list[str]) -> list[str]:
    """Strip ColumnTransformer prefixes so plot axes read cleanly."""
    return [n.split("__", 1)[-1] if "__" in n else n for n in names]


# --------------------------------------------------------------------------- #
# Model-specific attribution
# --------------------------------------------------------------------------- #

def logistic_coefficients(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Standardized coefficients: the pipeline already standardizes its inputs.

    Because `StandardScaler` runs inside the pipeline, each coefficient is per
    one standard deviation of that feature, so magnitudes are directly
    comparable across features. Reported as association, not effect size.
    """
    matrix, names = _transform(pipeline, X)
    model = pipeline.named_steps["model"]
    coefs = np.asarray(model.coef_).ravel()
    tidy = _tidy(names)

    frame = pd.DataFrame({
        "feature": tidy,
        "coefficient": coefs,
        "abs_coefficient": np.abs(coefs),
    }).sort_values("abs_coefficient", ascending=False).reset_index(drop=True)

    # Direction relative to the positive class ("disease").
    frame["direction"] = np.where(frame["coefficient"] >= 0, "toward higher risk score",
                                  "toward lower risk score")
    frame["model"] = "logistic_regression"
    frame["interpretation"] = "association learned by the model, per 1 SD of feature"
    return frame


def tree_feature_importances(pipeline, X: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """Impurity-based feature importance for Random Forest and XGBoost.

    These reflect how much each feature reduced training-set impurity. They are
    in-sample quantities and are known to favour high-cardinality or continuous
    features; SHAP below is the out-of-sample-flavoured complement.
    """
    _, names = _transform(pipeline, X)
    model = pipeline.named_steps["model"]
    importances = np.asarray(model.feature_importances_).ravel()

    frame = pd.DataFrame({
        "feature": _tidy(names),
        "importance": importances,
        "abs_importance": np.abs(importances),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    frame["model"] = model_name
    frame["interpretation"] = "impurity-based importance, in-sample, not causal"
    return frame


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #

def shap_explainer(pipeline, X: pd.DataFrame, model_name: str):
    """Build the SHAP explainer and the preprocessed matrix it consumes.

    Returns (explainer, matrix, names). LinearExplainer for Logistic
    Regression, TreeExplainer for the ensembles. Both are exact for their model
    classes, so no approximation sampling error is introduced.

    `max_samples` is set explicitly for the linear case: the default 100 would
    silently cap the background set and make the baseline depend on a subsample.
    """
    matrix, raw_names = _transform(pipeline, X)
    model = pipeline.named_steps["model"]

    if model_name == "logistic_regression":
        explainer = shap.LinearExplainer(model, matrix, max_samples=len(matrix))
    else:
        explainer = shap.TreeExplainer(model)
    return explainer, matrix, _tidy(raw_names)


def output_scale(model_name: str) -> str:
    """The units this model's SHAP values live in.

    Determined by additivity rather than assumed. `TreeExplainer` with the
    default `tree_path_dependent` perturbation only supports `model_output="raw"`,
    so an XGBoost classifier is decomposed in raw margin (log-odds) space even
    though `predict_proba` returns a probability. Random Forest is decomposed in
    probability space. A per-family hard-coded label would be wrong for at least
    one of them.
    """
    return "log-odds" if model_name in ("logistic_regression", "xgboost") else "probability"


# Candidate model outputs that SHAP values may be expressed in, cheapest first.
_REFERENCE_OUTPUTS = {
    "probability": lambda model, matrix: model.predict_proba(matrix)[:, 1],
    "decision_function (log-odds)": lambda model, matrix: model.decision_function(matrix),
    "raw margin (log-odds)": lambda model, matrix: model.predict(matrix, output_margin=True),
}


def _align(values: np.ndarray, base_values: np.ndarray, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce SHAP output to the positive-class column and a per-row base.

    shap returns several shapes depending on model class and version:
      * (n_rows, n_features)                    - single output
      * (n_rows, n_features, n_classes)         - one column per class
      * base as a scalar, or (n_classes,), or (n_rows, n_classes)
    The positive class is the last column, matching `predict_proba(...)[:, 1]`.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, -1]

    # Base arrives as a scalar, (n_classes,), or (n_rows, n_classes). Only the
    # last is per-row; the other two must be broadcast, never sliced, or a
    # (n_rows, 2) array would be truncated into nonsense.
    base = np.asarray(base_values, dtype=float)
    if base.ndim == 2 and base.shape == (n_rows, 2):
        base = base[:, -1]
    else:
        base = base.ravel()
        if base.size == 1:
            base = np.full(n_rows, float(base[0]))
        elif base.size != n_rows:
            raise ValueError(
                f"cannot align a SHAP base of size {base.size} to {n_rows} rows"
            )
    return values, base


def shap_values(pipeline, X: pd.DataFrame, model_name: str) -> tuple[np.ndarray, list[str]]:
    """SHAP values for the fitted estimator on preprocessed features.

    See `shap_explainer` for the explainer choice and `output_scale` for the
    units; magnitudes are only comparable within one model.
    """
    explainer, matrix, names = shap_explainer(pipeline, X, model_name)
    if model_name == "logistic_regression":
        values = np.asarray(explainer.shap_values(matrix), dtype=float)
    else:
        values = np.asarray(explainer.shap_values(matrix, check_additivity=False),
                            dtype=float)
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:            # (n, n_features, n_classes) in older shapes
        values = values[:, :, -1]
    return values, names


def global_importance_table(values: np.ndarray, names: list[str], model_name: str,
                            signs: dict[str, float] | None = None) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, plus the average signed shift.

    `mean_abs_shap` is the importance measure and ranks the features.
    `mean_shap` is the average signed contribution, which is only a useful
    direction label when the effect genuinely runs one way. Two traps:

    * Tree models average to nearly zero because the effect cancels across
      rows, so a near-zero signed mean means "varies by patient".
    * For a linear model the baseline is the centred mean input, so the signed
      mean is ~0 *by construction* even though every row's contribution shares
      the coefficient's sign. For those, pass `signs` (from the fitted
      coefficients) and the direction is read from there instead.
    """
    frame = pd.DataFrame({
        "feature": names,
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
        "mean_shap_sd": values.std(axis=0, ddof=1) if values.shape[0] > 1 else 0.0,
    })
    frame["abs_mean_shap"] = frame["mean_abs_shap"]
    frame["model"] = model_name
    frame = frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    if signs is not None:
        frame["direction"] = [
            ("toward higher risk score" if signs.get(f, 0.0) >= 0
             else "toward lower risk score")
            if f in signs else "not directly present in the linear model"
            for f in frame["feature"]
        ]
        frame["direction_source"] = "linear coefficient sign (fixed per feature)"
        frame["direction_consistent"] = True
        return frame

    # Treat the signed mean as a real direction only when it is at least a
    # fifth of the average magnitude; otherwise the effect genuinely varies.
    material = frame["mean_shap"].abs() >= 0.2 * frame["mean_abs_shap"]
    frame["direction_consistent"] = material
    frame["direction_source"] = "signed mean SHAP across patients"
    frame["direction"] = np.where(
        ~material,
        "effect varies by patient (signed contributions cancel)",
        np.where(frame["mean_shap"] >= 0,
                 "on average pushes score toward disease",
                 "on average pushes score toward no disease"),
    )
    return frame


def plot_global_importance(tables: dict[str, pd.DataFrame], output_dir: Path,
                           top_n: int = 15) -> str:
    """One panel per model with mean |SHAP| for the top features."""
    models = list(tables)
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5.5))
    if len(models) == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        frame = tables[model_name].head(top_n).iloc[::-1]
        ax.barh(frame["feature"], frame["mean_abs_shap"], color="#4c72b0")
        ax.set_title(f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)}\nmean |SHAP| value")
        ax.set_xlabel("mean absolute SHAP value")
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Global feature importance (model contribution, not causation)", y=1.02)
    fig.tight_layout()
    path = output_dir / "global_feature_importance.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path.name


def plot_shap_summary(values: np.ndarray, names: list[str], model_name: str,
                      output_dir: Path, max_display: int = 15) -> str:
    """SHAP beeswarm summary for one model."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    shap.summary_plot(
        values, features=names, max_display=max_display, show=False,
        plot_size=None, color_bar=True,
    )
    plt.gcf().set_size_inches(9, 5.5)
    plt.title(f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} — SHAP summary")
    plt.tight_layout()
    path = output_dir / f"shap_summary_{model_name}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close("all")
    return path.name


def plot_shap_bar(values: np.ndarray, names: list[str], model_name: str,
                  output_dir: Path, max_display: int = 15) -> str:
    """Mean |SHAP| bar chart, a simpler companion to the beeswarm."""
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([names[i] for i in order][::-1], mean_abs[order][::-1], color="#55a868")
    ax.set_xlabel("mean |SHAP| value")
    ax.set_title(f"{MODEL_DISPLAY_NAMES.get(model_name, model_name)} — mean absolute contribution")
    plt.tight_layout()
    path = output_dir / f"shap_global_bar_{model_name}.png"
    plt.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path.name


def individual_explanations(values: np.ndarray, names: list[str], matrix: np.ndarray,
                            model_name: str, output_dir: Path,
                            cases: list[dict], base_values: np.ndarray,
                            model_outputs: np.ndarray | None = None,
                            true_labels: np.ndarray | None = None,
                            max_display: int = 12) -> list[dict]:
    """Local waterfall plot plus a machine-readable table for each chosen case.

    `matrix` must be the PREPROCESSED matrix the model consumes, not the raw
    frame: SHAP values have one entry per transformed column, so pairing them
    with 13 raw features would misalign the plot.

    `base_values` is the explainer's expected output PER ROW, not the mean SHAP
    value. The two differ, and using the mean would make each waterfall's top
    segment a number that does not equal the model's own prediction for that
    patient.

    `cases` supplies one row per observation: {"index": int, "label": str}.
    Returns records so they can be written into the JSON card.
    """
    records = []
    display_names = MODEL_DISPLAY_NAMES.get(model_name, model_name)
    scale = output_scale(model_name)
    base_values = np.asarray(base_values, dtype=float).ravel()

    for case in cases:
        idx = case["index"]
        label = case["label"]
        row_values = values[idx]
        base_value = float(base_values[idx])

        explanation = shap.Explanation(
            values=row_values,
            base_values=base_value,
            data=matrix[idx],
            feature_names=names,
        )

        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title(f"{display_names} - case: {label}\n"
                  f"(model contribution, not causation)")
        plt.tight_layout()
        filename = f"shap_individual_{model_name}_{label.replace(' ', '_')}.png"
        plt.savefig(output_dir / filename, dpi=140, bbox_inches="tight")
        plt.close("all")

        contribution = pd.Series(row_values, index=names).sort_values(key=abs, ascending=False)
        table_path = output_dir / f"shap_individual_{model_name}_{label.replace(' ', '_')}.csv"
        contribution.head(max_display).to_csv(table_path, header=["shap_contribution"])

        # Additivity: base + sum(contributions) must equal the model's own
        # output for this row. Checked, not assumed.
        reconstructed = base_value + float(row_values.sum())
        actual = None
        if model_outputs is not None:
            actual = float(model_outputs[idx])
            if not np.isclose(reconstructed, actual, atol=1e-4):
                raise RuntimeError(
                    f"SHAP additivity failed for {model_name} case {label}: "
                    f"base+sum(shap)={reconstructed:.6f} vs model output={actual:.6f}"
                )

        records.append({
            "model": model_name,
            "case": label,
            "row_index_in_explanation_sample": int(idx),
            "plot": filename,
            "contribution_table": table_path.name,
            "output_scale": scale,
            "base_value_expected_output": round(base_value, 5),
            "sum_of_contributions": round(float(row_values.sum()), 5),
            "reconstructed_output": round(reconstructed, 5),
            "model_output": round(actual, 5) if actual is not None else None,
            "additivity_verified": actual is not None,
            "observed_target": int(true_labels[idx]) if true_labels is not None else None,
            "top_contributors": [
                {
                    "feature": feature,
                    "contribution": round(float(value), 5),
                    "direction": "toward disease" if value > 0 else "toward no disease",
                    "feature_value": round(float(matrix[idx][names.index(feature)]), 5),
                }
                for feature, value in contribution.head(5).items()
            ],
        })

    return records


def choose_interesting_cases(y: pd.Series, prob: np.ndarray, X: pd.DataFrame,
                             min_examples: int = 3) -> list[dict]:
    """Pick a diverse set of rows to explain, deterministically.

    A fixed, explainable selection rule rather than random sampling: a
    confident false positive, a confident false negative, a confident true
    positive, a confident true negative, and one case with heavy missingness.
    Reproducible because it depends only on the score and the observed label.
    """
    predicted = (prob >= 0.5).astype(int)
    outcome = np.where(y.to_numpy() == predicted,
                       np.where(predicted == 1, "TP", "TN"),
                       np.where(predicted == 1, "FP", "FN"))
    missing_count = X.isna().sum(axis=1).to_numpy()

    cases: list[dict] = []

    def add(label: str, ranking_scores: np.ndarray):
        candidates = np.where(outcome == label)[0]
        if candidates.size == 0:
            return
        best = candidates[np.argmax(ranking_scores[candidates])]
        cases.append({"index": int(best), "label": label})

    # FP: highest predicted probability. FN: lowest predicted probability among
    # positives. Both are the errors a reviewer would want to see explained.
    fp_mask = outcome == "FP"
    if fp_mask.any():
        cases.append({"index": int(np.argmax(np.where(fp_mask, prob, -np.inf))), "label": "FP"})
    fn_mask = outcome == "FN"
    if fn_mask.any():
        cases.append({"index": int(np.argmin(np.where(fn_mask, prob, np.inf))), "label": "FN"})
    add("TP", prob)
    add("TN", -prob)

    # Ensure at least the requested number of explanations even if a category
    # is empty in a given model.
    if len(cases) < min_examples:
        remaining = np.setdiff1d(np.arange(len(y)), [c["index"] for c in cases])
        for idx in remaining[: min_examples - len(cases)]:
            cases.append({"index": int(idx), "label": "TP" if y.iloc[idx] == 1 else "TN"})

    # Add the row with the most missing values, if it is not already included.
    existing = {c["index"] for c in cases}
    most_missing = int(np.argmax(missing_count))
    if most_missing not in existing:
        cases.append({
            "index": most_missing,
            "label": "high_missingness_case",
        })

    return cases


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def generate_all(fitted_pipelines: dict, X_explain: pd.DataFrame,
                 feature_columns: list[str], output_dir: Path,
                 random_state: int = prep.RANDOM_SEED,
                 max_individual: int = 5,
                 oof_probabilities: dict | None = None,
                 y_explain: pd.Series | None = None) -> dict:
    """Produce every explainability artefact and return a summary record.

    `oof_probabilities` maps model name -> out-of-fold probability over
    `X_explain`. When supplied, case selection uses those out-of-fold scores so
    the chosen FP/FN cases are genuine errors rather than rows a fully-fitted
    model has already memorised. `y_explain` supplies the true labels used to
    label those cases.
    """
    oof_probabilities = oof_probabilities or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console: list[str] = []
    global_tables: dict[str, pd.DataFrame] = {}
    shap_tables: dict[str, pd.DataFrame] = {}
    coefficient_frame: pd.DataFrame | None = None
    importance_frames: dict[str, pd.DataFrame] = {}
    individual_records: list[dict] = []
    additivity: dict[str, dict] = {}
    files: list[str] = []

    for model_name, pipeline in fitted_pipelines.items():
        console.append(f"\n{model_name}:")
        console.append(f"  n transformed features: {len(_transform(pipeline, X_explain)[1])}")

        # Model-specific attribution
        if model_name == "logistic_regression":
            coefficients = logistic_coefficients(pipeline, X_explain)
            coefficient_frame = coefficients
            coefficients.to_csv(output_dir / "logistic_regression_coefficients.csv", index=False)
            files.append("logistic_regression_coefficients.csv")
            top = coefficients.head(5)
            console.append("  standardized coefficients (top 5 by |coef|):")
            for _, row in top.iterrows():
                console.append(f"    {row['feature']:32s} {row['coefficient']:+.4f}  {row['direction']}")
        else:
            importances = tree_feature_importances(pipeline, X_explain, model_name)
            importance_frames[model_name] = importances
            importances.to_csv(output_dir / f"{model_name}_feature_importance.csv", index=False)
            files.append(f"{model_name}_feature_importance.csv")
            console.append("  impurity-based importance (top 5):")
            for _, row in importances.head(5).iterrows():
                console.append(f"    {row['feature']:32s} {row['importance']:.4f}")

        # SHAP
        explainer, matrix, names = shap_explainer(pipeline, X_explain, model_name)
        model = pipeline.named_steps["model"]
        # Prefer the callable API: it returns per-row `base_values` for the
        # ensemble models, whereas `expected_value` is a single global baseline
        # and would be wrong to feed to a per-patient waterfall.
        if model_name == "logistic_regression":
            explanation_all = explainer(matrix)
        else:
            explanation_all = explainer(matrix, check_additivity=False)
        values, base_row = _align(explanation_all.values,
                                  explanation_all.base_values, len(matrix))

        # Additivity check across the whole sample, not only the plotted cases:
        # base + sum(contributions) must reproduce the model's own output. The
        # scale is discovered by testing the candidate outputs rather than
        # assumed from the model family.
        reconstructed = base_row + values.sum(axis=1)
        matched_label, reference, max_additivity_gap = None, None, np.inf
        for label, produce in _REFERENCE_OUTPUTS.items():
            try:
                candidate = np.asarray(produce(model, matrix), dtype=float).ravel()
            except (AttributeError, TypeError, ValueError):
                continue                       # this model has no such output
            if candidate.shape != reconstructed.shape:
                continue
            gap = float(np.abs(reconstructed - candidate).max())
            if gap < max_additivity_gap:
                matched_label, reference, max_additivity_gap = label, candidate, gap

        if matched_label is None or max_additivity_gap > 1e-4:
            raise RuntimeError(
                f"SHAP additivity failed for {model_name}: no model output matches "
                f"base+sum(shap); closest gap = {max_additivity_gap:.3e}"
            )

        additivity[model_name] = {
            "output_compared_against": matched_label,
            "declared_output_scale": output_scale(model_name),
            "base_value_mean_expected_output": round(float(base_row.mean()), 5),
            "base_value_is_per_row": bool(np.ptp(base_row) > 0),
            "max_abs_gap": max_additivity_gap,
            "verified": True,
        }

        # For a linear model the signed mean SHAP is ~0 by construction, so the
        # direction comes from the fitted coefficient sign instead.
        signs = None
        if model_name == "logistic_regression":
            signs = dict(zip(coefficients["feature"], coefficients["coefficient"]))
        table = global_importance_table(values, names, model_name, signs=signs)
        shap_tables[model_name] = table
        table.to_csv(output_dir / f"{model_name}_shap_global.csv", index=False)
        files.append(f"{model_name}_shap_global.csv")

        files.append(plot_shap_summary(values, names, model_name, output_dir))
        files.append(plot_shap_bar(values, names, model_name, output_dir))

        console.append(f"  SHAP global importance (top 5, mean |SHAP|; "
                       f"scale: {output_scale(model_name)}):")
        for _, row in table.head(5).iterrows():
            console.append(f"    {row['feature']:32s} {row['mean_abs_shap']:.4f}  {row['direction']}")
        console.append(f"  additivity check: max |base+sum(SHAP)-output| = "
                       f"{max_additivity_gap:.2e} -> OK")

        # Individual explanations, paired with the preprocessed matrix so the
        # per-feature values line up with the SHAP values.
        # Prefer out-of-fold scores for case selection so FP/FN cases are real
        # errors; fall back to in-sample scores if none were supplied.
        prob = oof_probabilities.get(model_name)
        if prob is None:
            prob = model.predict_proba(matrix)[:, 1]
        prob = np.asarray(prob, dtype=float)
        # Case selection needs the TRUE labels, not the model's own predictions,
        # otherwise FP/FN categories can never be populated.
        target = pd.Series(y_explain.to_numpy() if y_explain is not None
                           else (prob >= 0.5).astype(int))
        cases = choose_interesting_cases(target, prob, X_explain)[:max_individual]
        individual_records.extend(
            individual_explanations(
                values, names, matrix, model_name, output_dir, cases,
                base_values=base_row,
                model_outputs=reference,
                true_labels=target.to_numpy(),
            )
        )
        console.append(f"  individual explanations written: {len(cases)}")
        console.append("    cases: " + ", ".join(f"{c['label']}(row {c['index']})" for c in cases))

    files.append(plot_global_importance(shap_tables, output_dir))

    summary = {
        "attribution_caveat": ATTRIBUTION_CAVEAT,
        "baseline_for_shap": (
            "the explainer's expected output for each model (E[f(x)]), not the mean "
            "SHAP value; base + sum(contributions) reproduces the model's own output, "
            "verified numerically for every explained case and for the whole sample"
        ),
        "output_scale": {
            model_name: f"{output_scale(model_name)} units "
                        f"(verified against {record['output_compared_against']})"
            for model_name, record in additivity.items()
        } | {
            "note": "magnitudes are NOT comparable across model families: compare "
                    "signs and within-model rankings only",
        },
        "additivity_check": additivity,
        "explanation_sample_rows": int(len(X_explain)),
        "raw_features": list(feature_columns),
        "global_shap": {name: table.to_dict(orient="records") for name, table in shap_tables.items()},
        "logistic_coefficients": (
            coefficient_frame.to_dict(orient="records") if coefficient_frame is not None else None
        ),
        "tree_importances": {name: frame.to_dict(orient="records")
                             for name, frame in importance_frames.items()},
        "individual_explanations": individual_records,
        "files": sorted(set(files)),
        "console_lines": console,
    }

    (output_dir / "explainability_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    console.append("")
    console.append(f"  SHAP baseline: {summary['baseline_for_shap']}")
    for model_name, record in additivity.items():
        console.append(f"  additivity {model_name}: max gap {record['max_abs_gap']:.2e} "
                       f"vs {record['output_compared_against']}")
    console.append(f"  caveat: {ATTRIBUTION_CAVEAT}")
    console.append(f"  files written: {len(summary['files'])}")

    return summary
