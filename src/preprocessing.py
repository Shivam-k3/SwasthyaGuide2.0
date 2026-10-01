"""
SWASTHYAGUIDE 2.0 — Heart Disease Risk module
Stage 2: preprocessing pipeline definition.

Design rules enforced here
--------------------------
1.  `data/heart_disease_uci.csv` is READ-ONLY. Nothing in this module writes to it.
    `load_dataset()` verifies the file hash and raises if the file was altered.
2.  Every preprocessing step lives INSIDE an sklearn `Pipeline` that wraps a
    `ColumnTransformer`. Imputers and encoders are therefore fitted on training
    data only — the test set can never influence them, and an artefact saved
    with `joblib` reproduces training-time preprocessing exactly at inference.
3.  `id` and `dataset` are NEVER predictive features. `dataset` is retained
    outside the feature set purely for cohort-aware splitting and reporting.
4.  The label rule is `num == 0 -> 0` (no disease), `num > 0 -> 1` (disease).
5.  Invalid physiological sentinels are converted to NaN by a transformer that
    lives inside the pipeline, so the rule also applies to unseen inference data.

Scope: research prototype for health RISK ASSESSMENT. Not a medical device and
not a diagnostic system.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Paths and reproducibility constants
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "heart_disease_uci.csv"
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_SEED = 42

# SHA-256 of the pristine source file. Recorded so any accidental edit to the
# raw data is detected immediately instead of silently changing results.
EXPECTED_SHA256 = "574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae"

# --------------------------------------------------------------------------- #
# Column roles
# --------------------------------------------------------------------------- #

ID_COL = "id"                 # identifier metadata — excluded from features
COHORT_COL = "dataset"        # source metadata — splitting/reporting only
RAW_TARGET_COL = "num"        # ordinal 0-4 as shipped
TARGET_COL = "target"         # binary label derived from `num`

#: Columns never used as predictive features.
EXCLUDED_FROM_FEATURES = [ID_COL, COHORT_COL, RAW_TARGET_COL, TARGET_COL]

#: Boolean-looking columns that arrive as the *strings* "True"/"False".
BOOLEAN_SOURCE_COLS = ["fbs", "exang"]

#: Numeric features. Median imputation + missingness indicator.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]

#: Categorical features. Explicit "Missing" level + one-hot encoding.
CATEGORICAL_FEATURES = ["sex", "cp", "restecg", "slope", "thal"]

#: Columns that contain missing values in the raw file and therefore justify a
#: missingness indicator. `age` is complete in the source data, so adding an
#: indicator for it would create an all-zero, constant column.
NUMERIC_FEATURES_WITH_MISSING = ["trestbps", "chol", "thalch", "oldpeak", "ca"]

#: Explicit category used for missing values in categorical features.
MISSING_CATEGORY = "Missing"

#: The full ordered model feature list, i.e. exactly what a caller must supply
#: at inference time. Fitted transformers append derived columns (imputed-missing
#: indicators and one-hot dummies) on top of these.
#: Order is fixed and alphabetical-within-group for reproducible feature ordering.
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_SOURCE_COLS

# --------------------------------------------------------------------------- #
# Invalid-value rules (Task 5)
# --------------------------------------------------------------------------- #
# A zero resting blood pressure, a zero serum cholesterol, or a *negative* ST
# depression are not physiologically plausible measurements. In this dataset
# `chol == 0` appears in 172 rows (all Switzerland / VA Long Beach) and reads as
# "not recorded" rather than "measured as zero". Such values must not be fed to a
# scaler or a tree as if they were real readings, so they are mapped to NaN.
#
# Applied as a function INSIDE the pipeline, so unseen inference data is cleaned
# by the same rule the model was trained on.
INVALID_VALUE_RULES: dict[str, dict] = {
    "chol": {"op": "==", "value": 0.0, "reason": "zero serum cholesterol is not measurable"},
    "trestbps": {"op": "==", "value": 0.0, "reason": "zero resting blood pressure is not measurable"},
    "oldpeak": {"op": "<", "value": 0.0, "reason": "ST depression is recorded as magnitude, never negative"},
}

FEATURE_DOC: dict[str, str] = {
    "age": "numeric - patient age in years",
    "trestbps": "numeric - resting blood pressure (mm Hg)",
    "chol": "numeric - serum cholesterol (mg/dl)",
    "thalch": "numeric - maximum heart rate achieved (bpm)",
    "oldpeak": "numeric - ST depression induced by exercise",
    "ca": "ordinal numeric - number of major vessels with narrowing, 0-3 (see CA_DECISION)",
    "sex": "categorical - patient sex",
    "cp": "categorical - chest pain type",
    "restecg": "categorical - resting ECG result",
    "slope": "categorical - peak exercise ST slope",
    "thal": "categorical - thalassemia test result",
    "fbs": "binary - fasting blood sugar above 120 mg/dl",
    "exang": "binary - exercise induced angina",
}

CA_DECISION = """
`ca` DECISION — treated as ORDINAL NUMERIC, not one-hot categorised
=================================================================

Inspected before deciding (not assumed). Evidence from data/heart_disease_uci.csv:

* Values are exactly {0.0, 1.0, 2.0, 3.0} — integral, and in the original UCI
  specification `ca` is a *count* of major vessels with significant narrowing.
* Risk is monotonic in the count:

      ca = 0 -> 26.5% disease   (n=181)
      ca = 1 -> 68.7% disease   (n= 67)
      ca = 2 -> 80.5% disease   (n= 41)
      ca = 3 -> 85.0% disease   (n= 20)

  Monotonic non-decreasing, so an ordinal treatment matches the data-generating
  process rather than fighting it.
* Rank correlation confirms a real ordering signal: Spearman(ca, target)
  rho = 0.485, p = 1.4e-19; Spearman(ca, num) rho = 0.528, p = 1.4e-23.

CONSEQUENCE, AND IT IS A REAL ONE:
66.4% of `ca` is missing, and 99% of the non-missing values come from Cleveland
alone (Hungary 3, Switzerland 5, VA Long Beach 2). The training-set median is
therefore 0.0, so median imputation writes "0 narrowed vessels" for every
missing row — a healthy-looking value — even though rows with missing `ca` are
*more* likely to be diseased (59.7% vs 46.6%). Imputing the median alone would
inject a systematic bias toward "no disease".

DECISION: keep `ca` ordinal (documented above) AND always attach a
`ca_missing` indicator through `SimpleImputer(add_indicator=True)`. The
indicator lets the model learn the elevated base rate attached to an unrecorded
`ca` and corrects the optimistic direction of the median fill. This is exactly
the case where "median imputation" alone would be wrong, and it is why the
indicator is not optional.

Reviewed for leakage, not just fit: `ca` correlates 0.516 with `num`. Both
describe vessel narrowing, so `ca` is genuinely informative rather than an
accidental copy of the label. It is retained, and this correlation is disclosed
as a limitation in the README.
"""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def file_sha256(path: Path = DATA_PATH) -> str:
    """SHA-256 of a file, used to prove the raw data is untouched."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path: Path = DATA_PATH, verify_hash: bool = True) -> pd.DataFrame:
    """Load the raw CSV read-only and verify its checksum.

    The returned frame is a fresh copy; nothing writes back to disk.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Expected data/heart_disease_uci.csv."
        )
    if verify_hash:
        actual = file_sha256(path)
        if actual != EXPECTED_SHA256:
            raise ValueError(
                "Raw dataset checksum mismatch.\n"
                f"  expected {EXPECTED_SHA256}\n"
                f"  actual   {actual}\n"
                "The source CSV must stay unchanged. Restore it before continuing."
            )
    return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# Frame-level preparation (applied before splitting; split-safe because it uses
# no fitted statistics and no information from any other row)
# --------------------------------------------------------------------------- #

def add_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create the binary label in memory: `num == 0 -> 0`, `num > 0 -> 1`.

    `num` is left untouched so the ordinal target remains available.
    """
    if RAW_TARGET_COL not in df.columns:
        raise KeyError(f"Expected target column '{RAW_TARGET_COL}' in the dataset.")
    out = df.copy()
    out[TARGET_COL] = out[RAW_TARGET_COL].gt(0).astype(int)
    return out


def coerce_boolean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the text columns `fbs` and `exang` into real binary values.

    In the source file these hold the strings "True"/"False" inside object
    columns. They become float 1.0 / 0.0 with NaN preserved, so the imputer can
    treat genuine absence differently from a recorded "False".
    """
    out = df.copy()
    for col in BOOLEAN_SOURCE_COLS:
        if col not in out.columns:
            raise KeyError(f"Expected boolean column '{col}' in the dataset.")
        raw = out[col]
        as_text = raw.astype("string").str.strip().str.lower()
        mapped = as_text.map({"true": 1.0, "false": 0.0})
        # Anything unrecognised stays NaN rather than being silently coerced.
        out[col] = pd.to_numeric(mapped, errors="coerce").astype(float)
    return out


def build_analysis_frame(verify_hash: bool = True) -> pd.DataFrame:
    """Raw CSV -> analysis frame with target and booleans set up.

    Invalid sentinel values are deliberately NOT removed here. They are handled
    inside the pipeline (see `make_invalid_value_masker`) so that the same rule
    applies to inference data.
    """
    df = load_dataset(verify_hash=verify_hash)
    df = add_binary_target(df)
    df = coerce_boolean_columns(df)
    return df


def split_xy(df: pd.DataFrame):
    """Return (X, y, cohorts) using only the approved feature columns."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise KeyError(f"Dataset is missing expected feature columns: {missing}")
    X = df.loc[:, FEATURE_COLUMNS].copy()
    y = df[TARGET_COL].copy()
    cohorts = df[COHORT_COL].copy()
    return X, y, cohorts


# --------------------------------------------------------------------------- #
# Pipeline components
# --------------------------------------------------------------------------- #

class InvalidValueMasker(BaseEstimator, TransformerMixin):
    """Map physiologically invalid sentinels to NaN, in place of real measurements.

    Applies the rules in `INVALID_VALUE_RULES` to the named columns and leaves
    everything else untouched. Rows are never dropped.

    Implemented as a class rather than a closure so the fitted pipeline can be
    serialised with joblib: pickle stores the class by reference, whereas a
    locally-defined function cannot be pickled at all.
    """

    def __init__(self, columns: list[str], rules: dict[str, dict] | None = None):
        # Stored exactly as given: sklearn's `clone` requires that __init__
        # parameters are not rebuilt or copied inside the constructor.
        self.columns = columns
        self.rules = rules

    def _resolved_rules(self) -> dict[str, dict]:
        return INVALID_VALUE_RULES if self.rules is None else self.rules

    def fit(self, X, y=None):  # noqa: N803 - sklearn naming
        return self

    def transform(self, X):  # noqa: N803 - sklearn naming
        arr = np.asarray(X, dtype=float).copy()
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        index = {col: i for i, col in enumerate(self.columns)}
        for col, rule in self._resolved_rules().items():
            if col not in index:
                continue
            j = index[col]
            col_values = arr[:, j]
            if rule["op"] == "==":
                arr[:, j] = np.where(col_values == rule["value"], np.nan, col_values)
            elif rule["op"] == "<":
                arr[:, j] = np.where(col_values < rule["value"], np.nan, col_values)
            elif rule["op"] == "<=":
                arr[:, j] = np.where(col_values <= rule["value"], np.nan, col_values)
            else:  # pragma: no cover - guards against a typo in INVALID_VALUE_RULES
                raise ValueError(f"Unsupported invalid-value op: {rule['op']!r}")
        return arr

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns, dtype=object)


class CategoricalMissingFiller(BaseEstimator, TransformerMixin):
    """Fill missing categorical values with an explicit `Missing` level.

    An explicit level keeps absence visible to the model instead of hiding it
    behind a mode fill. Dropping rows is not an option here: `thal` is 52.8% and
    `slope` 33.6% missing, so row deletion would destroy most of the dataset.

    A class rather than a closure so the pipeline stays picklable.
    """

    def __init__(self, missing_value: str = MISSING_CATEGORY):
        self.missing_value = missing_value

    def fit(self, X, y=None):  # noqa: N803 - sklearn naming
        return self

    def transform(self, X):  # noqa: N803 - sklearn naming
        out = X.copy()
        if isinstance(out, pd.DataFrame):
            for col in out.columns:
                out[col] = out[col].astype("object")
                out[col] = out[col].where(pd.notna(out[col]), self.missing_value)
            return out
        arr = np.asarray(out, dtype=object).copy()
        # pd.isna over object cells catches both None and float NaN.
        mask = pd.isna(arr)
        arr[mask] = self.missing_value
        return arr

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("CategoricalMissingFiller needs input feature names.")
        return np.asarray(list(input_features), dtype=object)


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    scale: bool = True,
) -> ColumnTransformer:
    """Build the ColumnTransformer used by every model.

    Three branches:
      * numeric  — continuous measurements plus the two binary flags `fbs` and
                   `exang`. Invalid sentinels become NaN, then median imputation
                   with a missingness indicator per column that has any missing
                   values in the fitted data, then optional scaling.
      * categorical — an explicit `Missing` level, then one-hot encoding.

    `fbs` and `exang` are handled in the numeric branch on purpose. They are
    genuinely binary, and once converted from text they are floats that can be
    median-imputed (filling a missing flag with the training majority) without
    inventing fake categories.

    Identical preprocessing is applied to all three baselines so the model
    comparison is fair. `StandardScaler` is included for all of them: it is
    required by Logistic Regression, and it is a monotone transform that
    partition-based trees are invariant to, so it does not disadvantage
    Random Forest or XGBoost while keeping one canonical pipeline.

    Every imputer is fitted on whatever data the pipeline is fitted with, which
    is why the pipeline must always be fitted on training data only.
    """
    if numeric_features is None:
        numeric_features = list(NUMERIC_FEATURES) + list(BOOLEAN_SOURCE_COLS)
    else:
        numeric_features = list(numeric_features)
    categorical_features = list(
        CATEGORICAL_FEATURES if categorical_features is None else categorical_features
    )

    numeric_steps = [
        ("mask_invalid_values", InvalidValueMasker(columns=list(numeric_features))),
        (
            "impute_median",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        ),
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical_steps = [
        ("fill_missing", CategoricalMissingFiller()),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",   # unseen level at inference -> all-zero row
                sparse_output=False,
                drop=None,                 # keep full dummies; regularisation handles redundancy
            ),
        ),
    ]

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            ("categorical", Pipeline(categorical_steps), categorical_features),
        ],
        remainder="drop",  # guarantees id / dataset / num can never reach a model
        verbose_feature_names_out=True,
    )


def build_pipeline(model, scale: bool = True) -> Pipeline:
    """Full preprocessing + model pipeline, ready to fit and to persist."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(scale=scale)),
            ("model", model),
        ]
    )


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    """Names of the columns a fitted pipeline actually produces.

    Longer than FEATURE_COLUMNS because imputed-missing indicators and one-hot
    dummies are appended. Recorded in the model card so an inference contract
    can be checked.
    """
    pre = pipeline.named_steps["preprocess"]
    return list(pre.get_feature_names_out())


# --------------------------------------------------------------------------- #
# Baseline model factories — deliberately un-tuned
# --------------------------------------------------------------------------- #
# Round default hyper-parameters only. No search, no tuning, no early stopping
# on the test set. The point of this stage is an honest baseline that later
# tuning work can be compared against.

def make_logistic_regression(random_state: int = RANDOM_SEED):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        class_weight=None,
        random_state=random_state,
    )


def make_random_forest(random_state: int = RANDOM_SEED, n_jobs: int = -1):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight=None,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def make_xgboost(random_state: int = RANDOM_SEED):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_lambda=1.0,
        eval_metric="logloss",
        tree_method="hist",
        random_state=random_state,
        n_jobs=1,      # single-threaded for bit-exact reproducibility
        verbosity=0,
    )


BASELINE_MODELS = {
    "logistic_regression": make_logistic_regression,
    "random_forest": make_random_forest,
    "xgboost": make_xgboost,
}
