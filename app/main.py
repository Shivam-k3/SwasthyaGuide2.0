"""
SWASTHYAGUIDE 2.0 - Heart Disease Risk module
Stage 4: the ASGI application.

Three endpoints, one loaded pipeline, no state between requests.

The pipeline is loaded in a `lifespan` handler rather than at import time, so a
missing or corrupt artifact produces a clean 503 from `/health` instead of an
import crash, and so tests can drive loading explicitly. The loaded model is
held on `app.state`, which is per-application: two app instances in one process
cannot see each other's model, and no prediction can depend on a previous
request.

If the artifact cannot be loaded, the process still starts and still answers
`/health`. That is deliberate: a health endpoint that cannot respond is useless
to whatever is monitoring it, and the failure needs to be visible rather than
silent.

Security controls live in `app/security.py` and are installed by
`create_app`. `app` is the module-level instance built from the process
environment; tests build their own instances with an explicit `SecuritySettings`
rather than mutating global state. Everything here is at the application
boundary - the frozen Stage 3 pipeline, its threshold and its preprocessing are
untouched by any of it.

Research prototype for health RISK ASSESSMENT. Not a medical device, not a
diagnostic system.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status

from app import model_service as service
from app import security
from app.schemas import (
    DISCLAIMER,
    HealthResponse,
    ModelInfoResponse,
    PatientRiskRequest,
    RiskPredictionResponse,
)
from app.security import SecuritySettings

logger = logging.getLogger("swasthyguide.api")

#: Rejected before any model is touched. One message for every cause, so a
#: caller learns that the service is not available without learning anything
#: about the server's filesystem or internals.
UNAVAILABLE_DETAIL = "The prediction model is currently unavailable. Please retry later."


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Load the persisted pipeline once, at startup. Never fits anything."""
    try:
        application.state.model = service.load_model()
        logger.info(
            "Serving model %s at threshold %s", application.state.model.name,
            application.state.model.threshold,
        )
    except service.ModelLoadError as exc:
        # Log the cause for the operator; return only the generic message.
        logger.error("Model unavailable: %s", exc)
        application.state.model = None
    try:
        yield
    finally:
        application.state.model = None


def create_app(
    settings: SecuritySettings | None = None,
    limiter: security.RateLimiter | None = None,
) -> FastAPI:
    """Build the application for one resolved security configuration.

    A factory rather than a single module-level object so that a production
    configuration (docs off, HSTS on, closed CORS) and a development
    configuration can coexist in one process - which is what makes the security
    tests possible without touching the developer's own environment.
    """
    resolved = settings or SecuritySettings.from_env()

    application = FastAPI(
        title="SWASTHYAGUIDE 2.0 - Heart Disease Risk API",
        version=service.API_VERSION,
        summary=(
            "Research risk-score inference. Not a medical device and not a diagnostic "
            "system. Outputs must never be used to diagnose, treat, or rule out any "
            "condition."
        ),
        description=(
            "## Disclaimer\n\n"
            "SWASTHYAGUIDE 2.0 is a **research prototype** for health **risk "
            "assessment**. It is **not** a medical device, **not** a diagnostic system, "
            "and has not been validated for clinical decision-making. A response never "
            "states that a person does or does not have a condition. Always consult a "
            "qualified healthcare professional.\n\n"
            "## How to read a prediction\n\n"
            "`risk_probability` is what this model estimated from the fields you "
            "supplied. It is not a diagnosis and not a probability of your future "
            "health. `expected_operating_point` reports how the served model actually "
            "behaved at the active threshold on the project's development data, so the "
            "false-alarm versus missed-alarm trade-off is visible rather than implied.\n\n"
            "## Design principle\n\n"
            "A false alarm is preferable to a missed alarm, so the served threshold "
            "favours sensitivity over specificity. This is a deployment decision and is "
            "configurable, not a clinical finding.\n\n"
            "## Missing values\n\n"
            "Omit any field you do not have. Missing values are imputed from training "
            "data with an explicit missingness indicator, exactly as the model was "
            "trained. Values that are not physiologically measurable - a cholesterol or "
            "resting blood pressure of 0, a negative ST depression - are accepted and "
            "treated as not recorded.\n\n"
            "## Operational\n\n"
            "Every response carries an `X-Request-ID` header. Quote it when reporting a "
            "problem; it is the correlation key in the server logs. Health inputs are "
            "never logged."
        ),
        lifespan=lifespan,
        # Documentation is environment-controlled, not deleted. In production it
        # is off by default and can be re-enabled explicitly; the schema itself
        # remains reachable in-process through `application.openapi()`.
        docs_url="/docs" if resolved.docs_enabled else None,
        redoc_url="/redoc" if resolved.docs_enabled else None,
        openapi_url="/openapi.json" if resolved.docs_enabled else None,
    )

    security.install_security(application, resolved, limiter)

    @application.get("/health", response_model=HealthResponse, summary="Liveness and model availability")
    def health(request: Request) -> HealthResponse:
        """Confirm the API is running and whether a model is loaded.

        Returns 503 when the model could not be loaded, which is the signal a
        deployment needs in order to stop routing traffic here.

        Deliberately minimal: status, model availability, the served operating
        point and the disclaimer. No filesystem paths, no model locations, no
        dependency inventory, no environment variables.
        """
        loaded = getattr(request.app.state, "model", None)
        if loaded is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=UNAVAILABLE_DETAIL
            )
        return HealthResponse(
            status="ok",
            api_stage=service.API_STAGE,
            api_version=service.API_VERSION,
            model_loaded=True,
            model=loaded.name,
            deployment_threshold=loaded.threshold,
            disclaimer=DISCLAIMER,
        )

    @application.get(
        "/model-info",
        response_model=ModelInfoResponse,
        response_model_exclude_none=True,
        summary="Served model metadata",
    )
    def model_info(request: Request) -> ModelInfoResponse:
        """Metadata for the served model: inputs, threshold, provenance, caveats.

        Safe public metadata only. Artifact references are repository-relative
        with no directory component, and the Stage 3 build-environment block is
        withheld unless an operator opts in, because it is an OS build string
        and a dependency inventory rather than research provenance.
        """
        return service.build_model_info(
            _require_model(request),
            include_build_environment=_build_environment_allowed(request),
        )

    @application.post("/predict", response_model=RiskPredictionResponse, summary="Score one patient")
    def predict(payload: PatientRiskRequest, request: Request) -> RiskPredictionResponse:
        """Score one patient.

        Inference is stateless: the response depends only on the fields in this
        request, never on previous requests, other patients, or batch position.
        """
        loaded = _require_model(request)
        try:
            return service.build_prediction_response(loaded, payload)
        except service.ModelLoadError as exc:
            logger.error("Prediction failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=UNAVAILABLE_DETAIL
            ) from exc

    return application


def _require_model(request: Request) -> service.LoadedModel:
    loaded = getattr(request.app.state, "model", None)
    if loaded is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=UNAVAILABLE_DETAIL
        )
    return loaded


def _build_environment_allowed(request: Request) -> bool:
    settings = getattr(request.app.state, "security_settings", None)
    if settings is not None:
        return bool(getattr(settings, "expose_build_environment", False))
    return False


#: The process-wide application, configured from the environment.
app = create_app()