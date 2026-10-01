"""
SWASTHYAGUIDE 2.0 — Stage 3 artifact integrity.

Stage 4 must not touch a single byte of the frozen Stage 1-3 output. This module
hashes `data/` and every file under `models/` and compares against a baseline
recorded outside the repository, so a write cannot hide inside the test run.

The baseline lives in the OS temp directory, never in `models/`, because adding
a file to `models/` would itself change the frozen inventory.

    %TEMP%\\opencode\\sw2_stage3_baseline.json

If the baseline is absent it is created from the current state, which is only
meaningful for a first run. The baseline used for the accepted Stage 4 work was
captured before any Stage 4 code existed; `python -m src.verify_stage3` (run in
`test_api.py`) independently re-checks the split, the checksum, the leakage
invariants and the artifact inventory, so a corrupted artifact cannot pass on
the strength of a stale hash file alone.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from src import preprocessing as prep

BASELINE_PATH = (
    Path(os.environ.get("TEMP", "/tmp")) / "opencode" / "sw2_stage3_baseline.json"
)

FROZEN_DIRECTORIES = ("data", "models")


def _fingerprint() -> dict[str, str]:
    manifest: dict[str, str] = {}
    for directory in FROZEN_DIRECTORIES:
        for path in sorted((prep.PROJECT_ROOT / directory).rglob("*")):
            if path.is_file():
                key = str(path.relative_to(prep.PROJECT_ROOT)).replace("\\", "/")
                manifest[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


@pytest.fixture(scope="module")
def baseline() -> dict[str, str]:
    if BASELINE_PATH.exists():
        return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest = _fingerprint()
    BASELINE_PATH.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return manifest


def test_baseline_covers_the_whole_frozen_inventory(baseline: dict[str, str]) -> None:
    current = _fingerprint()
    assert set(baseline) == set(current), "the frozen file set changed"
    assert "data/heart_disease_uci.csv" in baseline
    assert "models/stage3_report.json" in baseline
    assert "models/tuned/logistic_regression_pipeline.joblib" in baseline


def test_no_frozen_artifact_was_modified(baseline: dict[str, str]) -> None:
    """The central Stage 4 guarantee."""
    current = _fingerprint()
    changed = [k for k in baseline if baseline[k] != current.get(k)]
    added = sorted(set(current) - set(baseline))
    removed = sorted(set(baseline) - set(current))
    assert not changed, f"Stage 4 modified frozen artifacts: {changed}"
    assert not added, f"Stage 4 added files to frozen directories: {added}"
    assert not removed, f"Stage 4 removed frozen artifacts: {removed}"


def test_raw_dataset_checksum_matches_the_recorded_value() -> None:
    assert prep.file_sha256() == prep.EXPECTED_SHA256


@pytest.mark.parametrize(
    "artifact",
    [
        "models/stage3_report.json",
        "models/baseline_report.json",
        "models/calibration_metrics.csv",
        "models/calibration_curve.png",
        "models/threshold_analysis.csv",
        "models/holdout_final_metrics.csv",
        "models/tuned/logistic_regression_pipeline.joblib",
        "models/tuned/random_forest_pipeline.joblib",
        "models/tuned/xgboost_pipeline.joblib",
        "models/explainability/explainability_summary.json",
    ],
)
def test_key_stage3_artifacts_match_the_baseline(
    baseline: dict[str, str], artifact: str
) -> None:
    assert artifact in baseline, f"{artifact} is missing from the baseline"
    assert _fingerprint()[artifact] == baseline[artifact]


def test_explainability_file_count_is_unchanged() -> None:
    """Stage 3 recorded 44 explainability files; a rewrite would change this."""
    explainability = prep.MODELS_DIR / "explainability"
    assert len(list(explainability.iterdir())) == 44


def test_api_writes_nothing_into_the_repository() -> None:
    """No artifact is written by serving: the service only reads."""
    service = pytest.importorskip("app.model_service")
    loaded = service.load_model()
    before = _fingerprint()

    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as client:
        client.post(
            "/predict",
            json={
                "age": 57, "sex": "Male", "cp": "asymptomatic",
                "trestbps": 140.0, "chol": 0.0, "thalch": 150.0,
                "oldpeak": 1.5, "ca": None,
            },
        )
        client.get("/model-info")

    assert _fingerprint() == before
    assert loaded.pipeline is not None
