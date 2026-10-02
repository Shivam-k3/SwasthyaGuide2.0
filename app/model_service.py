"""
SWASTHYAGUIDE 2.0 — Heart Disease Risk module
Stage 4: model loading and inference.

The persisted sklearn `Pipeline` is the source of truth. Nothing here re-implements
imputation, sentinel masking, encoding or scaling; those steps live inside the
artifact exactly as they were fitted in Stage 2 and 3. The service loads that
artifact once at application startup and reuses it. It never calls `fit`.

Operating characteristics are READ from `models/stage3_report.json` rather than
recomputed. Reading costs nothing, touches no fold, and cannot become a new
modelling decision.

Threshold policy
----------------
SWASTHYAGUIDE's design rule is that a false alarm is preferable to a missed
alarm, so the served operating point favours recall over specificity. Stage 3
kept 0.50 for research reporting because no clinical threshold had been
specified; Stage 4 supplies that specification, which is what makes moving the
served threshold a legitimate deployment decision rather than a research one.

The default threshold is PER MODEL, in `DEPLOYMENT_THRESHOLDS`. Each entry is
that model's own measured operating point from the Stage 3 grid, not a number
tuned on one model and applied to another. Stage 3 evaluated the full 0.3-0.7
grid for all three models, so each has its own validated recall/specificity
trade-off; the figures differ by model and none is borrowed. Where a model has
no validated operating point, none is invented and the model is refused rather
than served at a borrowed or interpolated value.

Every threshold this module will accept is a point Stage 3 actually measured on
the 736-row out-of-fold development pool, and the operating characteristics
reported for the served model are that model's own measured values at that
threshold. Two guards keep that honest:

  * A model with no measured development record cannot be served at all.
  * `SWASTHYAGUIDE_THRESHOLD` is accepted only if it is a measured point for
    the SELECTED model. Setting Logistic Regression's 0.40 while serving Random
    Forest is therefore rejected at startup rather than applied silently.

The threshold is server-side only. `/predict` accepts no threshold field, so a
client cannot quietly disable the high-recall policy.

Nothing is interpolated, and the 184-row locked holdout is never read.

Research prototype for health RISK ASSESSMENT. Not a medical device, not a
diagnostic system.
"""

from __future__ import annotations

import functools
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import joblib

from src import preprocessing as prep
from app.schemas import (
    DISCLAIMER,
    ELEVATED,
    ELEVATED_MESSAGE,
    LOWER,
    LOWER_MESSAGE,
    ExpectedOperatingPoint,
    FeatureSpec,
    ModelInfoResponse,
    PatientRiskRequest,
    REQUIRED_FEATURES,
    RiskPredictionResponse,
    feature_kind,
)

logger = logging.getLogger("swasthyguide.api")

API_VERSION = "4.0.0"
API_STAGE = "4 - FastAPI serving layer"

STAGE3_REPORT_PATH = prep.MODELS_DIR / "stage3_report.json"

DEFAULT_MODEL = "logistic_regression"

#: Per-model deployment threshold, each a point Stage 3 measured for THAT model
#: on the 736-row out-of-fold development pool (`models/threshold_analysis.csv`).
#:
#: 0.40 is each model's own measured operating point, not Logistic Regression's
#: borrowed: Stage 3 evaluated the full 0.3-0.7 grid for all three models, so each
#: entry below is that model's validated recall/specificity trade-off. Applying
#: one model's tuned number to another would fabricate validation that never
#: happened. The evidence differs per model and is reported per response.
#:
#: Each value moves the default toward the project's high-recall rule within
#: existing evidence. None is interpolated, and the 184-row holdout was not used
#: to choose any of them.
DEPLOYMENT_THRESHOLDS: dict[str, float] = {
    # recall 0.897 / specificity 0.736 / F1 0.850 (vs 0.845 / 0.799 / 0.842 at 0.50)
    "logistic_regression": 0.40,
    # recall 0.929 / specificity 0.672 / F1 0.847 (vs 0.862 / 0.781 / 0.846 at 0.50)
    "random_forest": 0.40,
    # recall 0.892 / specificity 0.708 / F1 0.838 (vs 0.845 / 0.809 / 0.845 at 0.50)
    "xgboost": 0.40,
}

THRESHOLD_ENV = "SWASTHYAGUIDE_THRESHOLD"
MODEL_ENV = "SWASTHYAGUIDE_MODEL"

#: Stage 3 selected pipelines are the ones intended for serving. The Stage 2
#: un-tuned baselines under `models/` are deliberately not offered.
SERVED_MODELS: dict[str, str] = {
    name: f"tuned/{name}_pipeline.joblib" for name in prep.BASELINE_MODELS
}

#: Disclosed per model, so /model-info cannot imply a ranking Stage 3 refused to
#: make, and so a deliberately unrecalibrated model carries its caveat with it.
CALIBRATION_NOTES: dict[str, str] = {
    "logistic_regression": (
        "Closest to ideal calibration of the three: slope 1.01 and intercept "
        "approximately 0. Not recalibrated, because it needed no correction."
    ),
    "random_forest": (
        "Under-dispersed: calibration slope 1.33, so its scores are not extreme "
        "enough, and its expected calibration error is roughly double the other "
        "models. Deliberately NOT recalibrated, because recalibrating after "
        "seeing those results would be a modelling decision made on the "
        "evaluation data. Treat its probabilities with that caveat."
    ),
    "xgboost": (
        "Close to ideal calibration (slope 1.04), marginally behind Logistic "
        "Regression on expected calibration error. Not recalibrated."
    ),
}

PIPELINE_VERSION_NOTE = (
    "Fitted in Stage 3 on the 736-row locked-split training portion. The 184-row "
    "holdout was evaluated once at the end of Stage 3 and was never used for "
    "fitting, tuning, threshold selection or model selection."
)


class ModelLoadError(RuntimeError):
    """The served model cannot be prepared.

    Messages are safe to return over HTTP: they never carry a filesystem path.
    """


@dataclass(frozen=True)
class LoadedModel:
    """A fitted pipeline plus the frozen operating point it is served at."""

    name: str
    pipeline: object
    threshold: float
    operating_point: ExpectedOperatingPoint
    model_version: str


# --------------------------------------------------------------------------- #
# Stage 3 record, read once
# --------------------------------------------------------------------------- #

@functools.lru_cache(maxsize=1)
def _stage3_record() -> dict:
    """The Stage 3 validation record, read once and cached.

    Every operating characteristic the API reports comes from here rather than
    being recomputed. Reading is free, touches no fold, and cannot become a new
    modelling decision.
    """
    if not STAGE3_REPORT_PATH.exists():
        raise ModelLoadError(
            "The Stage 3 validation record is missing, so no measured operating "
            "point can be cited. Refusing to serve an unverified threshold."
        )
    try:
        return json.loads(STAGE3_REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelLoadError("The Stage 3 validation record could not be read.") from exc


def _measured_grid(model_name: str) -> dict[float, dict]:
    """Threshold points Stage 3 actually measured for one model.

    Keyed by float, so a lookup either finds a measured point or fails. There is
    no interpolation path by construction.
    """
    records = _stage3_record().get("threshold_analysis", [])
    return {
        float(r["threshold"]): r
        for r in records
        if r["model"] == model_name and r.get("feature_variant") == "full"
    }


def _operating_point(model_name: str, threshold: float) -> ExpectedOperatingPoint:
    """Turn one measured grid record into the reported operating point."""
    grid = _measured_grid(model_name)
    if threshold not in grid:
        raise ModelLoadError(
            f"No measured development threshold for model {model_name!r} at {threshold}. "
            f"Measured points: {sorted(grid)}."
        )
    record = grid[threshold]
    tn, fp, fn, tp = (int(record[k]) for k in ("tn", "fp", "fn", "tp"))
    n_rows = tn + fp + fn + tp
    negatives, positives = tn + fp, fn + tp
    # A zero denominator means the rate was never measurable at this operating
    # point, so it reports NaN rather than a misleading 0 or 1.
    fpr = fp / negatives if negatives else float("nan")
    fnr = fn / positives if positives else float("nan")
    return ExpectedOperatingPoint(
        source=(
            f"Stage 3 out-of-fold predictions on the {n_rows}-row development pool. "
            "These describe the model's behaviour on that pool, not this patient. "
            "The locked holdout was not used."
        ),
        n_rows=n_rows,
        n_positive=positives,
        threshold=threshold,
        recall=float(record["sensitivity_recall"]),
        specificity=float(record["specificity"]),
        precision=float(record["precision"]),
        f1=float(record["f1"]),
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        roc_auc=float(record["roc_auc"]),
        pr_auc=float(record["pr_auc"]),
        tn=tn, fp=fp, fn=fn, tp=tp,
    )


def _selection_rationale(model_name: str) -> str:
    """State the deployment choice honestly, with numbers from the record."""
    record = _stage3_record()
    cv = {r["model"]: r for r in record.get("cv_metrics_tuned_overall", [])}
    calibration = {r["model"]: r for r in record.get("calibration", [])}

    if model_name == DEFAULT_MODEL:
        lead = (
            "Default serving choice. On the 25-fold development estimate it had the "
            "highest ROC-AUC and PR-AUC of the three and the best calibration, and it "
            "is the simplest of the three to interpret."
        )
    else:
        lead = (
            f"Selectable, not endorsed. Stage 3's paired per-fold comparison could not "
            f"separate {model_name!r} from the other models on discrimination after "
            f"tuning, so no ranking is claimed."
        )
    stats = cv.get(model_name, {})
    cal = calibration.get(model_name, {})
    ece = cal.get("expected_calibration_error", cal.get("ece", float("nan")))
    return (
        f"{lead} Development estimate: ROC-AUC {stats.get('roc_auc_mean', float('nan')):.4f}, "
        f"PR-AUC {stats.get('pr_auc_mean', float('nan')):.4f}, "
        f"Brier {cal.get('brier_score', float('nan')):.4f}, "
        f"ECE {ece:.4f}. "
        "This is a deployment choice made on the project's own validation evidence "
        "and its interpretability requirements. It is NOT proof that this model is "
        "universally better, and it does not transfer to an unseen population."
    )


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def configured_model_name() -> str:
    """Which model `MODEL_ENV` selects, validated against the servable set.

    Falls back to `DEFAULT_MODEL` when unset or blank. Raises `ModelLoadError`
    for an unknown name rather than silently serving the default, so a typo in
    configuration cannot go unnoticed.
    """
    name = os.environ.get(MODEL_ENV, DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if name not in SERVED_MODELS:
        raise ModelLoadError(
            f"{MODEL_ENV}={name!r} is not a servable model. "
            f"Choose one of: {', '.join(sorted(SERVED_MODELS))}."
        )
    return name


def configured_threshold(model_name: str) -> float:
    """The deployment threshold, checked against that model's measured points."""
    raw = os.environ.get(THRESHOLD_ENV)
    if raw is None or not raw.strip():
        if model_name not in DEPLOYMENT_THRESHOLDS:
            # No operating point was validated for this model, so none is invented
            # and none is borrowed from another model.
            raise ModelLoadError(
                f"Model {model_name!r} has no validated development threshold, so it "
                "cannot be served at any default. Serve a model Stage 3 threshold-"
                "analysed, or set a measured value explicitly."
            )
        threshold = DEPLOYMENT_THRESHOLDS[model_name]
    else:
        try:
            threshold = float(raw)
        except ValueError as exc:
            raise ModelLoadError(f"{THRESHOLD_ENV}={raw!r} is not a number.") from exc
        if not 0.0 < threshold < 1.0:
            raise ModelLoadError(f"{THRESHOLD_ENV}={threshold} is outside (0, 1).")

    # The guard that matters: the threshold must be a point Stage 3 measured for
    # the model actually being served, not merely a sensible-looking number.
    grid = _measured_grid(model_name)
    if not grid:
        raise ModelLoadError(
            f"Model {model_name!r} has no measured development threshold analysis, so "
            "it cannot be served. Select a model Stage 3 evaluated at a threshold."
        )
    if threshold not in grid:
        raise ModelLoadError(
            f"Threshold {threshold} was never measured for model {model_name!r} on "
            f"development data. Measured points: {sorted(grid)}. Refusing to serve an "
            "unverified operating point; use one of the measured values."
        )
    return threshold


# --------------------------------------------------------------------------- #
# Loading and inference
# --------------------------------------------------------------------------- #

def load_model(model_name: str | None = None) -> LoadedModel:
    """Load the persisted pipeline and pin its operating point.

    Raises `ModelLoadError` with a message safe to return over HTTP.
    """
    name = model_name or configured_model_name()
    threshold = configured_threshold(name)

    path = prep.MODELS_DIR / SERVED_MODELS[name]
    if not path.exists():
        raise ModelLoadError(
            f"The persisted pipeline for {name!r} is not present. "
            "Run `python -m src.validate_stage3` to produce it."
        )
    try:
        pipeline = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - any unpickling failure must be contained
        logger.exception("Failed to load the persisted pipeline for %s", name)
        raise ModelLoadError(
            f"The persisted pipeline for {name!r} could not be loaded."
        ) from exc

    if not hasattr(pipeline, "predict_proba"):
        raise ModelLoadError(
            f"The artifact for {name!r} does not expose predict_proba, so it is not a "
            "fitted classifier pipeline."
        )

    record = _stage3_record()
    model_version = (
        f"stage3-{record.get('generated_utc', 'unknown')}"
        f"/data-{record.get('data', {}).get('sha256', 'unknown')[:12]}"
    )
    return LoadedModel(
        name=name,
        pipeline=pipeline,
        threshold=threshold,
        operating_point=_operating_point(name, threshold),
        model_version=model_version,
    )


def predict_probability(loaded: LoadedModel, request: PatientRiskRequest) -> float:
    """Positive-class probability for one patient, using the persisted pipeline."""
    frame = request.to_pipeline_frame()
    try:
        probability = loaded.pipeline.predict_proba(frame)[0, 1]
    except (ValueError, TypeError) as exc:
        logger.exception("Inference failed for model %s", loaded.name)
        raise ModelLoadError(
            "The request could not be scored by the persisted pipeline."
        ) from exc
    return float(probability)


def build_prediction_response(
    loaded: LoadedModel, request: PatientRiskRequest
) -> RiskPredictionResponse:
    """Assemble the full prediction response for one patient.

    The classification is `probability >= threshold`, computed on the unrounded
    value. Rounding to six places affects only the reported figure, so two
    responses are byte-comparable without the threshold decision depending on
    presentation.

    Nothing here interprets the score clinically. `ELEVATED_MESSAGE` and
    `LOWER_MESSAGE` are deliberately phrased as research output, and a low score
    explicitly does not rule out any condition.
    """
    probability = predict_probability(loaded, request)
    elevated = probability >= loaded.threshold
    return RiskPredictionResponse(
        model=loaded.name,
        model_version=loaded.model_version,
        risk_probability=round(probability, 6),
        prediction=ELEVATED if elevated else LOWER,
        threshold=loaded.threshold,
        message=ELEVATED_MESSAGE if elevated else LOWER_MESSAGE,
        notes=request.input_notes(),
        expected_operating_point=loaded.operating_point,
        disclaimer=DISCLAIMER,
    )


def build_model_info(
    loaded: LoadedModel, *, include_build_environment: bool = False
) -> ModelInfoResponse:
    """Metadata for the served model, read from the artifact and Stage 3 record.

    `include_build_environment` withholds the Stage 3 `environment` block by
    default. It carries an OS build string and exact dependency versions: useful
    as a targeting aid for known CVEs, and it tells a caller nothing about the
    risk assessment itself. The block still exists in the frozen report; this
    only controls whether the application republishes it.

    `artifact` is a bare filename, not a path. The full repository-relative
    location is already stated in prose in `artifact_stage` and
    `threshold_source`, so nothing is lost, while the response stops pointing at
    where artifacts live on disk.
    """
    estimator = loaded.pipeline.named_steps["model"]
    return ModelInfoResponse(
        model=loaded.name,
        model_type=type(estimator).__name__,
        # Filename only. No directory, no absolute path, no filesystem root.
        artifact=Path(SERVED_MODELS[loaded.name]).name,
        artifact_stage="Stage 3 selected configuration (models/tuned/)",
        provenance=PIPELINE_VERSION_NOTE,
        deployment_threshold=loaded.threshold,
        threshold_source=(
            f"Measured for {loaded.name!r} on the Stage 3 out-of-fold development "
            "pool at this exact value, from that model's own row of "
            "models/threshold_analysis.csv. Not borrowed from another model, not "
            "interpolated. The locked holdout was not used."
        ),
        threshold_policy=(
            "Chosen to favour recall over specificity, because a false alarm is "
            "preferable to a missed alarm in this project. This is a deployment "
            "decision, not a research finding; Stage 3 retained 0.50 for reporting "
            "because no clinical threshold had been specified then."
        ),
        n_input_features=len(prep.FEATURE_COLUMNS),
        input_features=[
            FeatureSpec(
                name=name,
                kind=feature_kind(name),
                required=name in REQUIRED_FEATURES,
                description=prep.FEATURE_DOC.get(name, ""),
            )
            for name in prep.FEATURE_COLUMNS
        ],
        n_transformed_features=len(prep.transformed_feature_names(loaded.pipeline)),
        available_models=sorted(SERVED_MODELS),
        selection_rationale=_selection_rationale(loaded.name),
        calibration_note=CALIBRATION_NOTES[loaded.name],
        environment=(
            _stage3_record().get("environment", {}) if include_build_environment else None
        ),
        disclaimer=DISCLAIMER,
    )
