"""
SWASTHYAGUIDE 2.0 — Heart Disease Risk module
Stage 4: request and response schemas.

Validation contract
-------------------
Three rules decide every field, and none of them is invented here:

1. NAMES come from `src.preprocessing`. `FEATURE_COLUMNS` fixes the 13 input
   features and their order; `FEATURE_DOC` supplies the descriptions. The
   response frame is assembled in `FEATURE_COLUMNS` order so the API cannot
   drift from the column order the pipeline was fitted with.

2. CATEGORY LEVELS come from the fitted encoder, not from this file. The
   strings below are the `categories_` of the persisted `OneHotEncoder` minus
   the synthetic `Missing` level. `tests/test_api.py` asserts that equality
   against the real artifact, so a silent mismatch fails the suite instead of
   degrading into an all-zero one-hot row. Note the upstream UCI spelling
   "reversable defect" — it is the literal in the data and must match exactly.

3. REQUIREDNESS comes from the source data's own missingness. `age`, `sex` and
   `cp` are complete in all 920 rows, so they are required; the other ten
   columns carry missing values in the file and are optional, which keeps
   absence representable and lets the Stage 2 `Missing` level and
   `missingindicator` columns do their job.

Range rules are limited to what the feature physically is. `ca` is a count of
0-3 narrowed major vessels, so anything outside that is a wrong request rather
than an unusual patient. `oldpeak` deliberately has NO lower bound: a negative
value is a documented invalid-sentinel convention in this project, and
rejecting it here would silently change the Stage 2 preprocessing methodology.
`chol == 0` and `trestbps == 0` are likewise accepted, because the source file
uses them to mean "not recorded" in 172 and 1 rows respectively.

Research prototype for health RISK ASSESSMENT. Not a medical device, not a
diagnostic system.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StrictBool

from app.input_notes import describe_input_interpretations
from src import preprocessing as prep

# --------------------------------------------------------------------------- #
# Categorical levels, matching the fitted encoder's categories_ minus "Missing"
# --------------------------------------------------------------------------- #

SEX_LEVELS = ("Female", "Male")
CP_LEVELS = ("asymptomatic", "atypical angina", "non-anginal", "typical angina")
RESTECG_LEVELS = ("lv hypertrophy", "normal", "st-t abnormality")
SLOPE_LEVELS = ("downsloping", "flat", "upsloping")
THAL_LEVELS = ("fixed defect", "normal", "reversable defect")

#: Complete in all 920 source rows, so a request must supply them.
REQUIRED_FEATURES: tuple[str, ...] = ("age", "sex", "cp")

#: `ca` is a count, not a continuous measurement; every other numeric feature
#: is a reading on a continuous scale.
ORDINAL_NUMERIC_FEATURES: frozenset[str] = frozenset({"ca"})

CATEGORICAL_KIND = "categorical"
BINARY_KIND = "binary"
NUMERIC_KIND = "numeric"
ORDINAL_KIND = "ordinal_numeric"

FeatureKind = Literal["numeric", "ordinal_numeric", "categorical", "binary"]


def feature_kind(name: str) -> str:
    """Feature kind, derived from the Stage 1 column-role lists."""
    if name in prep.BOOLEAN_SOURCE_COLS:
        return BINARY_KIND
    if name in prep.CATEGORICAL_FEATURES:
        return CATEGORICAL_KIND
    if name in ORDINAL_NUMERIC_FEATURES:
        return ORDINAL_KIND
    return NUMERIC_KIND


# --------------------------------------------------------------------------- #
# Request
# --------------------------------------------------------------------------- #

class PatientRiskRequest(BaseModel):
    """One patient, expressed in the project's own feature conventions.

    `extra="forbid"` turns an unknown field into a 422 instead of a silently
    ignored key, which is the same class of protection as the strict
    category levels: a typo must not look like a clean request.
    """

    model_config = ConfigDict(extra="forbid")

    # Required — complete in the source data, so absence is not representable.
    age: int = Field(ge=1, le=120, description="Age in years.")
    sex: Literal["Female", "Male"]
    cp: Literal["asymptomatic", "atypical angina", "non-anginal", "typical angina"]

    # Optional numeric. `ge=0` still admits the 0 sentinels on purpose.
    trestbps: float | None = Field(default=None, ge=0, description="Resting blood pressure (mm Hg).")
    chol: float | None = Field(default=None, ge=0, description="Serum cholesterol (mg/dl).")
    thalach: float | None = Field(
        default=None,
        ge=0,
        # The dataset and the fitted pipeline both spell this column "thalch", a
        # long-standing upstream UCI typo, and the pipeline name must not change.
        # "thalach" is accepted as an alias so a caller using the correct
        # clinical spelling is not rejected. Both reach the same column.
        validation_alias=AliasChoices("thalch", "thalach"),
        description="Maximum heart rate achieved (bpm). Accepts 'thalch' or 'thalach'.",
    )
    oldpeak: float | None = Field(default=None, description="ST depression induced by exercise.")
    ca: int | None = Field(
        default=None, ge=0, le=3,
        description="Major vessels with significant narrowing, 0-3. "
                    "Angiographic; usually unavailable in a screening setting.",
    )

    # Optional categorical. None reaches the pipeline as a missing value and
    # becomes the explicit "Missing" level, exactly as in Stage 2.
    restecg: Literal["lv hypertrophy", "normal", "st-t abnormality"] | None = None
    slope: Literal["downsloping", "flat", "upsloping"] | None = None
    thal: Literal["fixed defect", "normal", "reversable defect"] | None = None

    # Optional binary flags. Sent as 1.0/0.0 to match what the pipeline was
    # fitted on. Deliberately NOT routed through
    # `prep.coerce_boolean_columns`: that function maps the text "True"/"False"
    # and is not idempotent on floats — it would turn 1.0 into NaN.
    #
    # `StrictBool` because Pydantic's lax mode accepts "yes", "on" and 1 as
    # True. A flag silently read as True flips a patient's risk input without
    # anyone noticing, which is the same failure mode as a misspelled category.
    fbs: StrictBool | None = Field(default=None, description="Fasting blood sugar above 120 mg/dl.")
    exang: StrictBool | None = Field(default=None, description="Exercise-induced angina.")

    def to_pipeline_frame(self) -> pd.DataFrame:
        """One row, columns in `FEATURE_COLUMNS` order, for the fitted pipeline.

        Missing values stay missing here. Imputation, sentinel masking and
        encoding all happen inside the persisted `Pipeline`; doing any of it
        here would duplicate Stage 2 logic and risk drifting from it.
        """
        record: dict[str, object] = {
            "age": float(self.age),
            "trestbps": self.trestbps,
            "chol": self.chol,
            "thalch": self.thalach,
            "oldpeak": self.oldpeak,
            "ca": None if self.ca is None else float(self.ca),
            "sex": self.sex,
            "cp": self.cp,
            "restecg": self.restecg,
            "slope": self.slope,
            "thal": self.thal,
            "fbs": None if self.fbs is None else float(self.fbs),
            "exang": None if self.exang is None else float(self.exang),
        }
        return pd.DataFrame([record], columns=list(prep.FEATURE_COLUMNS))

    def input_notes(self) -> list[str]:
        """Human-readable record of how ambiguous inputs were interpreted.

        Reports the decisions the pipeline will make on this patient's behalf, so
        a `chol` of 0 is visibly "not recorded" rather than looking like a real
        measurement of zero. The rule table is read from `prep`, so these notes
        cannot drift from the preprocessing they describe.
        """
        return describe_input_interpretations(
            self.to_pipeline_frame().iloc[0].to_dict()
        )


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #

DISCLAIMER = (
    "Research prototype for health risk assessment. Not a medical device, not a "
    "diagnostic system, and not validated for clinical decision-making. This "
    "output is a research risk score, not a diagnosis, and must not be used to "
    "diagnose, treat, or rule out any condition. Consult a qualified healthcare "
    "professional."
)

ELEVATED = "elevated_risk"
LOWER = "lower_predicted_risk"

ELEVATED_MESSAGE = (
    "Elevated predicted risk. Clinical evaluation may be appropriate. "
    "This is a research risk score, not a diagnosis."
)
LOWER_MESSAGE = (
    "Lower predicted risk according to this model. This does not rule out any "
    "condition; only a clinician can assess that."
)


class ExpectedOperatingPoint(BaseModel):
    """What the served model actually did at this threshold, on development data.

    Every number is read from the Stage 3 out-of-fold record for THIS model at
    THIS threshold. They describe the model's behaviour on the 736-row
    development pool, not the patient in the request, and are reported so the
    false-alarm / missed-alarm trade-off is visible rather than implied.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    n_rows: int
    n_positive: int
    threshold: float
    recall: float
    specificity: float
    precision: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    roc_auc: float
    pr_auc: float
    tn: int
    fp: int
    fn: int
    tp: int


class RiskPredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    model_version: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    prediction: Literal["elevated_risk", "lower_predicted_risk"]
    threshold: float
    message: str
    notes: list[str]
    expected_operating_point: ExpectedOperatingPoint
    disclaimer: str


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    api_stage: str
    api_version: str
    model_loaded: bool
    model: str
    deployment_threshold: float
    disclaimer: str


class FeatureSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: FeatureKind
    required: bool
    description: str


class ModelInfoResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    model_type: str
    artifact: str
    artifact_stage: str
    provenance: str
    deployment_threshold: float
    threshold_source: str
    threshold_policy: str
    n_input_features: int
    input_features: list[FeatureSpec]
    n_transformed_features: int
    available_models: list[str]
    selection_rationale: str
    calibration_note: str
    #: Stage 3 build environment. Omitted unless an operator opts in via
    #: `SWASTHYAGUIDE_EXPOSE_BUILD_ENV`: it is an OS build string and a
    #: dependency inventory, which is deployment fingerprinting rather than
    #: research provenance.
    environment: dict[str, str] | None = None
    disclaimer: str
