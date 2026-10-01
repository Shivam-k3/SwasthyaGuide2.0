"""Validation and reporting helpers for the API's own preprocessing conventions.

Every rule here mirrors something that already exists in `src.preprocessing`.
This module restates nothing: it reads the sentinel table from there, so a
change to preprocessing is reflected here without a second edit.

The point of these helpers is that the decision the pipeline will make is visible
in the response, rather than the request appearing to mean exactly what it says.
A caller who sends `chol = 0` should be able to see that the value was treated as
"not recorded", not silently scored as a real measurement of zero.
"""

from __future__ import annotations

import pandas as pd

from src import preprocessing as prep

#: Shown when an optional feature was omitted, so `null` is visibly a deliberate
#: choice rather than a silent gap.
MISSING_INPUT_NOTE = (
    "Not supplied, so imputed from training data with a missingness indicator: "
)

#: Prefix for a value the pipeline's sentinel rules will treat as not recorded.
IMPLAUSIBLE_INPUT_NOTE = "was treated as not recorded"


def _is_missing(value: object) -> bool:
    """True when a value is absent in any of the forms the pipeline sees.

    `pd.isna` handles `None` and `NaN`.
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _violates_sentinel_rule(value: float, rule: dict) -> bool:
    """Does one value breach one rule from `prep.INVALID_VALUE_RULES`?

    The rule carries its own comparison operator (`==` for a sentinel such as
    zero cholesterol, `<` for a value that is physically impossible such as
    negative ST depression), so the meaning stays defined in one place.
    """
    if rule["op"] == "==":
        return value == rule["value"]
    return value < rule["value"]


def describe_input_interpretations(values: dict) -> list[str]:
    """Explain how the pipeline will interpret each ambiguous input.

    Returns notes describing two things a caller could otherwise not see:

    1. Values that breach a documented invalid-sentinel rule (`chol == 0`,
       `trestbps == 0`, negative `oldpeak`), which the pipeline masks to NaN and
       imputes with a missingness indicator.
    2. Optional features that were omitted entirely, which are imputed the same
       way.

    Nothing here decides anything. The persisted pipeline performs the actual
    masking and imputation; these notes report what it is going to do, using the
    same rule table, so the response is honest about the interpretation.
    """
    notes: list[str] = []

    for column, rule in prep.INVALID_VALUE_RULES.items():
        if column not in values:
            continue
        value = values[column]
        if _is_missing(value):
            continue
        if _violates_sentinel_rule(float(value), rule):
            notes.append(
                f"{column}={value:g} {IMPLAUSIBLE_INPUT_NOTE} "
                f"({rule['reason']}) and imputed with a missingness indicator."
            )

    omitted = [
        column
        for column in prep.FEATURE_COLUMNS
        if column in values and _is_missing(values[column])
    ]
    if omitted:
        notes.append(f"{MISSING_INPUT_NOTE}{', '.join(omitted)}.")

    return notes
