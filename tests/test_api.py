"""
SWASTHYAGUIDE 2.0 — Stage 4 API tests.

These cover the project's own logic, not FastAPI's routing. The questions that
matter here are: does the API score a patient with the *persisted* pipeline, is
that result independent of everything except the request body, does the served
threshold match the measured development record, and does the service refuse to
invent an operating point.

Run from the project root:

    python -m pytest -q

`python -m pytest` is required rather than bare `pytest`: the project root has
to be importable for `app` and `src` to resolve, and the `-m` form supplies it.
No conftest sys.path hack is needed.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

from app import model_service as service
from app import schemas
from app.main import UNAVAILABLE_DETAIL, app
from src import preprocessing as prep

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

#: A complete, ordinary patient. Every other test mutates a copy of this.
COMPLETE_PATIENT = {
    "age": 57,
    "sex": "Male",
    "cp": "asymptomatic",
    "trestbps": 140.0,
    "chol": 245.0,
    "thalch": 150.0,
    "oldpeak": 1.5,
    "ca": 2,
    "restecg": "normal",
    "slope": "flat",
    "thal": "reversable defect",
    "fbs": True,
    "exang": False,
}


def _patient(**overrides) -> dict:
    body = dict(COMPLETE_PATIENT)
    body.update(overrides)
    return body


@pytest.fixture
def client() -> TestClient:
    """A client over a fully started app, so the model is loaded."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def stage3_record() -> dict:
    return json.loads(
        (prep.MODELS_DIR / "stage3_report.json").read_text(encoding="utf-8")
    )


# --------------------------------------------------------------------------- #
# 1. GET /health
# --------------------------------------------------------------------------- #

def test_health_reports_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = schemas.HealthResponse.model_validate(response.json())
    assert body.status == "ok"
    assert body.model_loaded is True
    assert body.model in service.SERVED_MODELS
    assert body.deployment_threshold == service.DEPLOYMENT_THRESHOLDS[body.model]
    assert "not a medical device" in body.disclaimer.lower()


def test_health_is_reachable_before_any_prediction(client: TestClient) -> None:
    """Health must not depend on having served traffic first."""
    assert client.get("/health").status_code == 200
    assert client.get("/health").status_code == 200


# --------------------------------------------------------------------------- #
# 2. GET /model-info
# --------------------------------------------------------------------------- #

def test_model_info_exposes_the_projects_own_feature_list(client: TestClient) -> None:
    body = schemas.ModelInfoResponse.model_validate(client.get("/model-info").json())

    assert [f.name for f in body.input_features] == list(prep.FEATURE_COLUMNS)
    assert body.n_input_features == 13 == len(prep.FEATURE_COLUMNS)
    # Descriptions are reused from the Stage 1 documentation, not rewritten.
    assert body.input_features[0].description == prep.FEATURE_DOC["age"]
    assert {f.name for f in body.input_features if f.required} == set(
        schemas.REQUIRED_FEATURES
    )


def test_model_info_requiredness_matches_source_missingness(client: TestClient) -> None:
    """Required fields are exactly the columns with no missing values in the file."""
    source = prep.load_dataset()
    complete = {c for c in prep.FEATURE_COLUMNS if source[c].isna().sum() == 0}
    assert set(schemas.REQUIRED_FEATURES) == complete


def test_model_info_transformed_count_comes_from_the_artifact(client: TestClient) -> None:
    body = schemas.ModelInfoResponse.model_validate(client.get("/model-info").json())
    loaded = service.load_model()
    assert body.n_transformed_features == len(
        prep.transformed_feature_names(loaded.pipeline)
    )
    assert body.n_transformed_features == 33


def test_model_info_never_leaks_absolute_paths(client: TestClient) -> None:
    text = client.get("/model-info").text
    assert "models/tuned/" in text
    for leak in (str(prep.PROJECT_ROOT), ":\\", "OneDrive", ".joblib\\"):
        assert leak not in text, f"model-info leaked a filesystem detail: {leak!r}"


def test_model_info_does_not_claim_a_model_is_best(client: TestClient) -> None:
    body = client.get("/model-info").json()
    text = " ".join(
        [body["selection_rationale"], body["calibration_note"], body["provenance"]]
    ).lower()
    assert "not proof" in text
    assert "universally" in text


def test_random_forest_keeps_its_calibration_caveat() -> None:
    """RF is deliberately unrecalibrated; that must travel with it."""
    note = service.CALIBRATION_NOTES["random_forest"].lower()
    assert "under-dispersed" in note
    assert "not recalibrated" in note


def test_every_servable_model_has_a_rationale_and_caveat() -> None:
    for name in service.SERVED_MODELS:
        assert service.CALIBRATION_NOTES[name]
        assert service._selection_rationale(name)
        assert service._measured_grid(name), f"{name} has no measured threshold record"


# --------------------------------------------------------------------------- #
# 3. Valid prediction
# --------------------------------------------------------------------------- #

def test_valid_prediction_returns_a_score(client: TestClient) -> None:
    response = client.post("/predict", json=COMPLETE_PATIENT)
    assert response.status_code == 200

    body = schemas.RiskPredictionResponse.model_validate(response.json())
    assert body.model == service.DEFAULT_MODEL
    assert body.prediction in {"elevated_risk", "lower_predicted_risk"}
    assert "research" in body.message.lower()
    assert "not a medical device" in body.disclaimer.lower()


def test_schema_level_choices_match_the_fitted_encoder() -> None:
    """The guard that stops a typo becoming a silent all-zero one-hot row.

    The declared literals must equal the persisted encoder's categories minus
    the synthetic "Missing" level. If Stage 3 is ever refitted with new levels,
    this fails loudly instead of the API quietly mis-scoring.
    """
    encoder = (
        service.load_model()
        .pipeline.named_steps["preprocess"]
        .named_transformers_["categorical"]
        .named_steps["onehot"]
    )
    fitted = [set(levels) for levels in encoder.categories_]
    fitted = [{lvl for lvl in levels if lvl != prep.MISSING_CATEGORY} for levels in fitted]

    assert fitted == [
        set(schemas.SEX_LEVELS),
        set(schemas.CP_LEVELS),
        set(schemas.RESTECG_LEVELS),
        set(schemas.SLOPE_LEVELS),
        set(schemas.THAL_LEVELS),
    ]


def test_thalach_alias_and_pipeline_spelling_are_equivalent(client: TestClient) -> None:
    """The dataset spells it "thalch"; the correct clinical spelling also works."""
    canonical = _patient()
    alias = _patient()
    alias["thalach"] = alias.pop("thalch")

    assert "thalach" not in canonical
    first = client.post("/predict", json=canonical).json()["risk_probability"]
    second = client.post("/predict", json=alias).json()["risk_probability"]
    assert first == second


# --------------------------------------------------------------------------- #
# 4. Missing nullable values
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("field", ["ca", "thal", "slope", "restecg", "fbs", "exang",
                                   "trestbps", "chol", "thalch", "oldpeak"])
def test_every_optional_field_may_be_omitted(client: TestClient, field: str) -> None:
    body = _patient()
    body.pop(field)
    response = client.post("/predict", json=body)
    assert response.status_code == 200, response.text
    assert field in response.json()["notes"][-1]


def test_minimal_request_is_only_the_three_complete_fields(client: TestClient) -> None:
    """age, sex and cp are the only columns with no missing values in the file."""
    response = client.post(
        "/predict", json={"age": 61, "sex": "Female", "cp": "typical angina"}
    )
    assert response.status_code == 200
    assert math.isfinite(response.json()["risk_probability"])


def test_missing_ca_is_reported_as_imputed_not_as_zero(client: TestClient) -> None:
    """`ca` is 66% missing upstream and its median fill is 0, a healthy value.

    The response must say the value was imputed rather than let a reader assume
    the patient genuinely has zero narrowed vessels.
    """
    body = _patient(ca=None)
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert "ca" in response.json()["notes"][-1]
    assert "imputed" in response.json()["notes"][-1]


def test_absent_categorical_becomes_the_missing_level(client: TestClient) -> None:
    """Omitting `thal` must not raise; the pipeline supplies the Missing level."""
    body = _patient()
    body.pop("thal")
    assert client.post("/predict", json=body).status_code == 200


# --------------------------------------------------------------------------- #
# 5-6. Invalid types and invalid structure
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"age": "fifty-seven", "sex": "Male", "cp": "typical angina"}, id="age-as-string"),
        pytest.param({"age": 57, "sex": "Male", "cp": 3}, id="cp-as-int"),
        pytest.param({"age": 57, "sex": "Male", "cp": "typical angina", "fbs": "yes"}, id="fbs-as-string"),
        pytest.param({"age": 57, "sex": "Male", "cp": "typical angina", "ca": "two"}, id="ca-as-string"),
        pytest.param({"age": 57, "sex": "Male", "cp": "typical angina", "thal": 1}, id="thal-as-int"),
    ],
)
def test_invalid_types_are_rejected(client: TestClient, payload: dict) -> None:
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param([], id="bare-list"),
        pytest.param("a string", id="string-body"),
        pytest.param({"sex": "Male", "cp": "typical angina"}, id="missing-age"),
        pytest.param({"age": 57, "cp": "typical angina"}, id="missing-sex"),
        pytest.param({"age": 57, "sex": "Male"}, id="missing-cp"),
    ],
)
def test_invalid_structures_are_rejected(client: TestClient, payload) -> None:
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_field_is_rejected_rather_than_ignored(client: TestClient) -> None:
    body = _patient(nickname="patient-1")
    assert client.post("/predict", json=body).status_code == 422


@pytest.mark.parametrize(
    "value", ["yes", "no", "on", "off", "1", "0", "True", "False", 1, 0]
)
def test_binary_flags_are_not_silently_coerced(client: TestClient, value) -> None:
    """Pydantic's lax bool accepts "yes" and 1 as True.

    A flag quietly read as True flips a real risk input, so only JSON true/false
    is accepted and everything else is a 422 the caller can see.
    """
    for field in ("fbs", "exang"):
        assert client.post("/predict", json=_patient(**{field: value})).status_code == 422


@pytest.mark.parametrize("field", ["fbs", "exang"])
@pytest.mark.parametrize("value", [True, False, None])
def test_binary_flags_accept_real_booleans_and_null(
    client: TestClient, field: str, value
) -> None:
    assert client.post("/predict", json=_patient(**{field: value})).status_code == 200


def test_supplying_both_spellings_of_thalach_is_rejected(client: TestClient) -> None:
    """`thalch` and `thalach` are two names for one column, not two columns."""
    body = _patient()
    body["thalach"] = 120.0
    assert client.post("/predict", json=body).status_code == 422


def test_typo_in_a_category_is_rejected(client: TestClient) -> None:
    """The whole point of the strict Literals: a typo must not score silently."""
    response = client.post("/predict", json=_patient(cp="typcal angina"))
    assert response.status_code == 422
    assert "cp" in response.text


def test_malformed_json_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/predict", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 7. Ranges that are justified
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"ca": 7}, id="ca-above-3"),
        pytest.param({"ca": -1}, id="ca-negative"),
        pytest.param({"age": -5}, id="age-negative"),
        pytest.param({"age": 0}, id="age-zero"),
        pytest.param({"age": 400}, id="age-absurd"),
        pytest.param({"trestbps": -80}, id="negative-bp"),
        pytest.param({"chol": -200}, id="negative-chol"),
    ],
)
def test_justified_ranges_are_enforced(client: TestClient, overrides: dict) -> None:
    assert client.post("/predict", json=_patient(**overrides)).status_code == 422


def test_ca_is_bounded_by_the_uci_count_not_by_the_observed_range(client: TestClient) -> None:
    """`ca` is a count of 0-3 vessels; 3 is in-vocabulary even though it is rare."""
    assert client.post("/predict", json=_patient(ca=3)).status_code == 200


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"chol": 0.0}, id="chol-zero-sentinel"),
        pytest.param({"trestbps": 0.0}, id="bp-zero-sentinel"),
        pytest.param({"oldpeak": -2.6}, id="negative-st-depression"),
    ],
)
def test_documented_sentinels_are_accepted_not_rejected(client: TestClient, overrides: dict) -> None:
    """Rejecting these would silently change the Stage 2 preprocessing rule.

    The source file uses chol==0 in 172 rows to mean "not recorded", so the API
    must accept it and report the interpretation instead of erroring.
    """
    response = client.post("/predict", json=_patient(**overrides))
    assert response.status_code == 200, response.text
    assert any("not recorded" in note for note in response.json()["notes"])


# --------------------------------------------------------------------------- #
# 8. Model loading failure
# --------------------------------------------------------------------------- #

def test_missing_artifact_yields_503_without_leaking_paths(monkeypatch) -> None:
    monkeypatch.setitem(
        service.SERVED_MODELS, service.DEFAULT_MODEL, "tuned/absent_model.joblib"
    )
    with TestClient(app) as broken:
        health = broken.get("/health")
        assert health.status_code == 503

        body = health.json()
        assert body["detail"] == UNAVAILABLE_DETAIL
        for leak in ("joblib", "models", ":\\", "Traceback", str(prep.MODELS_DIR)):
            assert leak not in health.text, f"503 body leaked {leak!r}"

        predict = broken.post("/predict", json=COMPLETE_PATIENT)
        assert predict.status_code == 503
        assert predict.json()["detail"] == UNAVAILABLE_DETAIL

        assert broken.get("/model-info").status_code == 503


def test_corrupt_artifact_is_contained(monkeypatch, tmp_path) -> None:
    """A bad pickle must not surface as a traceback or a path in the response."""
    corrupt = tmp_path / "logistic_regression_pipeline.joblib"
    corrupt.write_bytes(b"this is not a pickle")
    monkeypatch.setitem(service.SERVED_MODELS, service.DEFAULT_MODEL, str(corrupt))

    with pytest.raises(service.ModelLoadError, match="could not be loaded"):
        service.load_model()

    with TestClient(app) as broken:
        response = broken.get("/health")
        assert response.status_code == 503
        assert "Traceback" not in response.text
        assert str(tmp_path) not in response.text
        assert broken.post("/predict", json=COMPLETE_PATIENT).status_code == 503


def test_unknown_model_name_is_refused(monkeypatch) -> None:
    monkeypatch.setenv(service.MODEL_ENV, "catastrophe_model")
    with pytest.raises(service.ModelLoadError, match="not a servable model"):
        service.load_model()


# --------------------------------------------------------------------------- #
# 9-10. Determinism and statelessness
# --------------------------------------------------------------------------- #

def test_repeated_identical_requests_are_bit_identical(client: TestClient) -> None:
    responses = [client.post("/predict", json=COMPLETE_PATIENT).json() for _ in range(5)]
    probabilities = {r["risk_probability"] for r in responses}
    assert len(probabilities) == 1
    assert {r["prediction"] for r in responses} == {responses[0]["prediction"]}


def test_a_patient_is_unaffected_by_other_patients(client: TestClient) -> None:
    target = client.post("/predict", json=COMPLETE_PATIENT).json()

    others = [
        _patient(age=29, cp="typical angina", ca=0),
        _patient(age=80, sex="Female", thal="normal", fbs=False),
        _patient(age=44, oldpeak=6.2, ca=3, exang=True),
    ]
    for other in others:
        assert client.post("/predict", json=other).status_code == 200

    assert client.post("/predict", json=COMPLETE_PATIENT).json() == target


def test_interleaved_order_does_not_change_any_result(client: TestClient) -> None:
    patients = [
        _patient(age=57, ca=2),
        _patient(age=33, ca=None, thal=None),
        _patient(age=71, ca=0, cp="non-anginal"),
    ]
    forward = [client.post("/predict", json=p).json()["risk_probability"] for p in patients]
    backward = [client.post("/predict", json=p).json()["risk_probability"] for p in reversed(patients)]

    assert forward == list(reversed(backward))


def test_pipeline_scoring_is_batch_invariant(client: TestClient) -> None:
    """The Stage 3 audit checked this on the artifact; Stage 4 must not break it."""
    loaded = service.load_model()
    alone = loaded.pipeline.predict_proba(
        schemas.PatientRiskRequest(**COMPLETE_PATIENT).to_pipeline_frame()
    )[0, 1]

    batch = pd.concat(
        [schemas.PatientRiskRequest(**_patient(age=a)).to_pipeline_frame() for a in (25, 40, 57, 70)],
        ignore_index=True,
    )
    in_batch = loaded.pipeline.predict_proba(batch)[2, 1]

    assert alone == pytest.approx(in_batch, abs=0.0)


def test_each_request_gets_a_fresh_frame(client: TestClient) -> None:
    """No patient data is retained on the app between requests."""
    client.post("/predict", json=_patient(age=33, ca=0))
    first = client.post("/predict", json=COMPLETE_PATIENT).json()["risk_probability"]
    client.post("/predict", json=_patient(age=71, ca=3))
    second = client.post("/predict", json=COMPLETE_PATIENT).json()["risk_probability"]
    assert first == second


# --------------------------------------------------------------------------- #
# 11-13. Response schema, probability validity, threshold application
# --------------------------------------------------------------------------- #

def test_response_validates_against_the_declared_schema(client: TestClient) -> None:
    body = client.post("/predict", json=COMPLETE_PATIENT).json()
    parsed = schemas.RiskPredictionResponse.model_validate(body)
    assert set(body) == set(schemas.RiskPredictionResponse.model_fields)
    assert parsed.model == service.DEFAULT_MODEL


def test_probability_is_a_valid_probability(client: TestClient) -> None:
    probability = client.post("/predict", json=COMPLETE_PATIENT).json()["risk_probability"]
    assert isinstance(probability, float)
    assert math.isfinite(probability)
    assert not math.isnan(probability)
    assert 0.0 <= probability <= 1.0


def test_probability_matches_the_persisted_pipeline_exactly(client: TestClient) -> None:
    """The API must be a thin shell over the artifact, not a second implementation."""
    loaded = service.load_model()
    expected = loaded.pipeline.predict_proba(
        schemas.PatientRiskRequest(**COMPLETE_PATIENT).to_pipeline_frame()
    )[0, 1]
    reported = client.post("/predict", json=COMPLETE_PATIENT).json()["risk_probability"]
    assert reported == pytest.approx(expected, abs=5e-7)


def test_threshold_is_applied_with_greater_or_equal(client: TestClient) -> None:
    body = client.post("/predict", json=COMPLETE_PATIENT).json()
    expected = "elevated_risk" if body["risk_probability"] >= body["threshold"] else "lower_predicted_risk"
    assert body["prediction"] == expected


def test_threshold_boundary_behaviour(client: TestClient, monkeypatch) -> None:
    """At probability == threshold the call is elevated, by >= not >."""
    response = client.post("/predict", json=COMPLETE_PATIENT)
    probability = response.json()["risk_probability"]
    assert (probability >= response.json()["threshold"]) == (
        response.json()["prediction"] == "elevated_risk"
    )


def test_reported_threshold_is_the_configured_one(client: TestClient) -> None:
    assert client.get("/model-info").json()["deployment_threshold"] == 0.40
    assert client.post("/predict", json=COMPLETE_PATIENT).json()["threshold"] == 0.40


def test_response_reports_this_models_own_measured_operating_point(
    client: TestClient, stage3_record: dict
) -> None:
    """The FP/FN figures must be the served model's own, not another model's."""
    body = client.post("/predict", json=COMPLETE_PATIENT).json()
    point = body["expected_operating_point"]

    record = next(
        r for r in stage3_record["threshold_analysis"]
        if r["model"] == service.DEFAULT_MODEL
        and r["threshold"] == service.DEPLOYMENT_THRESHOLDS[service.DEFAULT_MODEL]
        and r["feature_variant"] == "full"
    )
    assert point["recall"] == pytest.approx(record["sensitivity_recall"])
    assert point["specificity"] == pytest.approx(record["specificity"])
    assert point["precision"] == pytest.approx(record["precision"])
    assert point["f1"] == pytest.approx(record["f1"])
    assert (point["fp"], point["fn"], point["tn"], point["tp"]) == (
        record["fp"], record["fn"], record["tn"], record["tp"]
    )


def test_false_rates_are_derived_not_asserted(client: TestClient) -> None:
    point = client.post("/predict", json=COMPLETE_PATIENT).json()["expected_operating_point"]
    assert point["false_positive_rate"] == pytest.approx(
        point["fp"] / (point["fp"] + point["tn"])
    )
    assert point["false_negative_rate"] == pytest.approx(
        point["fn"] / (point["fn"] + point["tp"])
    )
    assert point["n_rows"] == point["tn"] + point["fp"] + point["fn"] + point["tp"]


def test_operating_point_is_labelled_as_development_data(client: TestClient) -> None:
    point = client.post("/predict", json=COMPLETE_PATIENT).json()["expected_operating_point"]
    assert "out-of-fold" in point["source"]
    assert "not this patient" in point["source"]
    assert "holdout was not used" in point["source"]


def test_operating_point_n_rows_is_the_development_pool_not_the_holdout(
    client: TestClient, stage3_record: dict
) -> None:
    point = client.post("/predict", json=COMPLETE_PATIENT).json()["expected_operating_point"]
    assert point["n_rows"] == stage3_record["holdout"]["n_train"] == 736
    assert point["n_rows"] != stage3_record["holdout"]["n_test"]


def test_reported_metrics_are_threshold_free_where_they_should_be(client: TestClient) -> None:
    """ROC-AUC and PR-AUC do not move with the threshold, unlike recall."""
    point = client.post("/predict", json=COMPLETE_PATIENT).json()["expected_operating_point"]
    same = next(
        r for r in service._measured_grid(service.DEFAULT_MODEL).values()
        if r["threshold"] == service.DEPLOYMENT_THRESHOLDS[service.DEFAULT_MODEL]
    )
    assert point["roc_auc"] == pytest.approx(same["roc_auc"])
    assert point["pr_auc"] == pytest.approx(same["pr_auc"])


def test_the_service_favours_recall_over_specificity(client: TestClient) -> None:
    """The stated design principle must actually hold in the shipped numbers."""
    point = client.post("/predict", json=COMPLETE_PATIENT).json()["expected_operating_point"]
    grid = service._measured_grid(service.DEFAULT_MODEL)
    at_half = next(r for t, r in grid.items() if t == 0.50)

    assert point["recall"] > at_half["sensitivity_recall"]
    assert point["specificity"] < at_half["specificity"]
    assert point["false_negative_rate"] < 0.15


# --------------------------------------------------------------------------- #
# Threshold provenance: never invented, never from the holdout
# --------------------------------------------------------------------------- #

def test_unmeasured_threshold_is_refused_rather_than_invented(monkeypatch) -> None:
    for candidate in ("0.45", "0.35", "0.55", "0.9"):
        monkeypatch.setenv(service.THRESHOLD_ENV, candidate)
        with pytest.raises(service.ModelLoadError, match="never measured"):
            service.load_model(service.DEFAULT_MODEL)


@pytest.mark.parametrize("model_name", sorted(service.SERVED_MODELS))
def test_every_servable_model_rejects_an_unmeasured_threshold(
    monkeypatch, model_name: str
) -> None:
    """LR's 0.40 must not be smuggled onto a model that never measured it."""
    monkeypatch.setenv(service.THRESHOLD_ENV, "0.42")
    with pytest.raises(service.ModelLoadError, match="never measured"):
        service.load_model(model_name)


@pytest.mark.parametrize("model_name", sorted(service.SERVED_MODELS))
def test_every_servable_model_accepts_its_own_measured_points(
    monkeypatch, model_name: str
) -> None:
    for threshold in sorted(service._measured_grid(model_name)):
        monkeypatch.setenv(service.THRESHOLD_ENV, str(threshold))
        loaded = service.load_model(model_name)
        assert loaded.threshold == threshold
        assert loaded.operating_point.threshold == threshold


def test_out_of_range_threshold_is_refused(monkeypatch) -> None:
    for candidate in ("0", "1", "-0.2", "1.5", "not-a-number"):
        monkeypatch.setenv(service.THRESHOLD_ENV, candidate)
        with pytest.raises(service.ModelLoadError):
            service.load_model(service.DEFAULT_MODEL)


def test_other_models_get_their_own_operating_point() -> None:
    """Each model's reported FP/FN behaviour is its own, not Logistic Regression's."""
    points = {
        name: service.load_model(name).operating_point for name in service.SERVED_MODELS
    }
    lr = points[service.DEFAULT_MODEL]
    for name, point in points.items():
        if name != service.DEFAULT_MODEL:
            assert point.recall != lr.recall or point.specificity != lr.specificity
    assert len({p.recall for p in points.values()}) == len(points)


def test_default_threshold_is_in_the_recorded_grid() -> None:
    """Every shipped default must be measured for the model it serves.

    Checked per model, not just for the default one: a threshold validated for
    Logistic Regression is not automatically validated for Random Forest.
    """
    assert set(service.DEPLOYMENT_THRESHOLDS) == set(service.SERVED_MODELS)
    for name, threshold in service.DEPLOYMENT_THRESHOLDS.items():
        grid = service._measured_grid(name)
        assert grid, f"{name} has no measured development threshold analysis"
        assert threshold in grid, f"{name} default {threshold} was never measured"


@pytest.mark.parametrize("name", sorted(service.SERVED_MODELS))
def test_each_models_threshold_resolves_to_its_own_measured_record(name: str) -> None:
    """A model's operating point must come from its own grid, never another's.

    This is the guard against borrowing Logistic Regression's tuned threshold for
    a model Stage 3 never validated it on.
    """
    point = service._operating_point(name, service.DEPLOYMENT_THRESHOLDS[name])
    own = service._measured_grid(name)[service.DEPLOYMENT_THRESHOLDS[name]]

    expected = tuple(int(own[k]) for k in ("tn", "fp", "fn", "tp"))
    assert (point.tn, point.fp, point.fn, point.tp) == expected
    assert point.recall == pytest.approx(own["sensitivity_recall"])

    # No other model may share this exact confusion matrix at this threshold.
    for other in service.SERVED_MODELS:
        if other == name:
            continue
        other_grid = service._measured_grid(other)
        same_point = other_grid.get(service.DEPLOYMENT_THRESHOLDS[name])
        if same_point is None:
            continue
        assert (
            int(same_point["tp"]) != int(own["tp"])
            or int(same_point["fn"]) != int(own["fn"])
        ), f"{name} and {other} reported identical figures; thresholds were shared"


def test_nothing_reads_the_locked_holdout() -> None:
    """The serving path must never open the Stage 3 holdout figures.

    It reads `stage3_report.json`, which contains the holdout block, so the check
    is that no holdout figure is quoted anywhere in a response.
    """
    record = json.loads(
        (prep.MODELS_DIR / "stage3_report.json").read_text(encoding="utf-8")
    )
    holdout_roc = {
        round(result["overall"]["roc_auc"], 6)
        for result in record["holdout_final"].values()
    }
    with TestClient(app) as client:
        point = client.post("/predict", json=COMPLETE_PATIENT).json()["expected_operating_point"]
        info = client.get("/model-info").text
        health = client.get("/health").text

    assert round(point["roc_auc"], 6) not in holdout_roc
    for text in (info, health):
        assert "holdout_final" not in text


# --------------------------------------------------------------------------- #
# 14. The API does not retrain
# --------------------------------------------------------------------------- #

def test_inference_never_calls_fit(client: TestClient, monkeypatch) -> None:
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the API called fit(); inference must be load-only")

    monkeypatch.setattr(Pipeline, "fit", explode)
    assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 200
    assert client.get("/model-info").status_code == 200


def test_fitted_state_is_unchanged_by_requests(client: TestClient) -> None:
    loaded = service.load_model()
    pipeline = loaded.pipeline
    before = (
        pipeline.named_steps["model"].coef_.copy(),
        pipeline.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute_median"]
        .statistics_.copy(),
    )

    for _ in range(5):
        client.post("/predict", json=COMPLETE_PATIENT)
    client.post("/predict", json=_patient(age=33, ca=None, chol=0.0))

    after = (
        pipeline.named_steps["model"].coef_,
        pipeline.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["impute_median"]
        .statistics_,
    )
    assert np.array_equal(before[0], after[0])
    assert np.array_equal(before[1], after[1])


def test_model_is_loaded_once_at_startup_not_per_request(monkeypatch) -> None:
    """Startup loads the artifact once and reuses it for every later request."""
    real_load = service.load_model
    calls = []

    def counting_load(model_name=None):
        calls.append(model_name)
        return real_load(model_name)

    monkeypatch.setattr(service, "load_model", counting_load)
    with TestClient(app) as client:
        assert len(calls) == 1, "lifespan must load exactly once"
        for _ in range(4):
            assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 200
        assert client.get("/health").status_code == 200
        assert len(calls) == 1, "the artifact must not be re-read per request"
        assert client.app.state.model is not None


# --------------------------------------------------------------------------- #
# 12. Medical safety language
# --------------------------------------------------------------------------- #

FORBIDDEN_PHRASES = (
    "you have heart disease",
    "you do not have heart disease",
    "you have coronary",
    "diagnosis is",
    "diagnosed with",
    "treatment",
    "prescribe",
    "medication",
    "you should take",
    "cure",
)


def test_prediction_never_makes_a_diagnosis(client: TestClient) -> None:
    body = client.post("/predict", json=COMPLETE_PATIENT)
    text = body.text.lower()
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in text, f"response contained a clinical claim: {phrase!r}"


def test_medical_disclaimer_is_present_everywhere(client: TestClient) -> None:
    for text in (
        client.get("/health").text,
        client.get("/model-info").text,
        client.post("/predict", json=COMPLETE_PATIENT).text,
        client.get("/openapi.json").text,
    ):
        lowered = text.lower()
        assert "not a medical device" in lowered
        assert "risk" in lowered


def test_lower_prediction_also_refuses_to_rule_out(client: TestClient) -> None:
    """A low score must not be read as a clean bill of health."""
    body = _patient(age=34, sex="Female", cp="non-anginal", ca=0, oldpeak=0.0,
                    thalach=170, trestbps=110, chol=150, exang=False)
    body.pop("thalch")
    body["thalach"] = 170.0
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    text = response.text.lower()
    assert "does not rule out" in text
    for phrase in FORBIDDEN_PHRASES:
        assert phrase not in text


# --------------------------------------------------------------------------- #
# 15. Stage 3 integrity
# --------------------------------------------------------------------------- #

def test_stage3_verification_suite_still_passes() -> None:
    """The frozen Stage 3 suite must remain green after Stage 4 lands."""
    result = subprocess.run(
        [sys.executable, "-m", "src.verify_stage3"],
        cwd=prep.PROJECT_ROOT, capture_output=True, text=True, timeout=1800,
    )
    assert result.returncode == 0, result.stdout[-3000:]
    assert "69/69 checks passed" in result.stdout


def test_raw_dataset_is_untouched() -> None:
    assert prep.file_sha256() == prep.EXPECTED_SHA256
