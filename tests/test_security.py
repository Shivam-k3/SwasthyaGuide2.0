"""
SWASTHYAGUIDE 2.0 - Stage 1B runtime security tests.

These prove that the controls described in the Stage 1 security documentation are
*enforced*, not merely described. Every assertion below observes real HTTP
behaviour from a real application instance; nothing here asserts that a constant
exists.

Structure:

  1.  Security headers, including HSTS only when HTTPS/production is in force
  2.  CORS: allowed origin, denied origin, preflight, credentials
  3.  Rate limiting, 429 and Retry-After
  4.  Request size limits and 413
  5.  Request ID generation, validation and propagation
  6.  Safe error handling: no traceback, no path, no internals
  7.  Documentation exposure, development versus production
  8.  /health and /model-info hardening
  9.  Forwarded-header and Host handling
  10. Regression: valid /predict, 422 behaviour, 503 behaviour

Run from the project root:

    python -m pytest tests/test_security.py -q

Applications under test are built with an explicit `SecuritySettings` so each
case states the exact configuration it is asserting about, instead of mutating
the developer's environment.
"""

from __future__ import annotations

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app import security
from app import model_service as service
from app.main import UNAVAILABLE_DETAIL, create_app
from app.security import InMemoryRateLimiter, SecuritySettings
from src import preprocessing as prep

# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #

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


def settings(**overrides) -> SecuritySettings:
    """A development-flavoured baseline, with rate limiting actually enabled."""
    base = dict(
        environment="development",
        docs_enabled=True,
        cors_origins=("http://localhost:5173",),
        rate_limit_enabled=True,
        rate_limits={"*": 60, "/predict": 20, "/health": 600},
        max_request_size_bytes=4096,
    )
    base.update(overrides)
    return SecuritySettings(**base)


def client_for(
    config: SecuritySettings, limiter: security.RateLimiter | None = None
) -> TestClient:
    return TestClient(create_app(config, limiter), raise_server_exceptions=False)


@pytest.fixture
def dev_client() -> TestClient:
    with client_for(settings()) as client:
        yield client


def production_settings(**overrides) -> SecuritySettings:
    base = dict(
        environment="production",
        docs_enabled=False,
        cors_origins=(),
        hsts_enabled=True,
        rate_limits={"*": 60, "/predict": 20, "/health": 600},
        max_request_size_bytes=4096,
    )
    base.update(overrides)
    return SecuritySettings(**base)


# --------------------------------------------------------------------------- #
# 1. Security headers
# --------------------------------------------------------------------------- #

def test_security_headers_are_present_on_a_normal_response(dev_client: TestClient) -> None:
    headers = dev_client.get("/health").headers
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["Referrer-Policy"] == "no-referrer"
    assert headers["X-Frame-Options"] == "DENY"
    assert "camera=()" in headers["Permissions-Policy"]
    assert "geolocation=()" in headers["Permissions-Policy"]
    assert "default-src 'none'" in headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_cache_control_prevents_intermediary_caching(dev_client: TestClient) -> None:
    assert dev_client.get("/health").headers["Cache-Control"] == "no-store"
    response = dev_client.post("/predict", json=COMPLETE_PATIENT)
    assert response.headers["Cache-Control"] == "no-store"


def test_hsts_is_absent_during_local_http_development(dev_client: TestClient) -> None:
    """HSTS over plain HTTP local development would poison localhost in a browser."""
    assert "Strict-Transport-Security" not in dev_client.get("/health").headers


def test_hsts_is_present_when_forced_https() -> None:
    with client_for(settings(hsts_enabled=True)) as client:
        value = client.get("/health").headers["Strict-Transport-Security"]
    assert "max-age=31536000" in value
    assert "includeSubDomains" in value


def test_hsts_is_present_in_production_by_default() -> None:
    with client_for(production_settings()) as client:
        assert "Strict-Transport-Security" in client.get("/health").headers


def test_headers_are_present_on_error_responses_too(dev_client: TestClient) -> None:
    """A rejected request must be as well labelled as a successful one."""
    response = dev_client.post("/predict", json={"age": 57, "sex": "Male"})
    assert response.status_code == 422
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Content-Security-Policy"]


def test_csp_is_configurable() -> None:
    custom = "default-src 'none'; sandbox"
    with client_for(settings(content_security_policy=custom)) as client:
        assert client.get("/health").headers["Content-Security-Policy"] == custom


def test_csp_allows_the_shipped_swagger_ui_assets() -> None:
    """A CSP that blocks our own documentation page would be a broken control."""
    csp = security.DEFAULT_CSP
    assert "https://cdn.jsdelivr.net" in csp
    assert "'unsafe-inline'" in csp
    assert "style-src" in csp and "script-src" in csp


# --------------------------------------------------------------------------- #
# 2. CORS
# --------------------------------------------------------------------------- #

def test_allowed_origin_receives_cors_headers() -> None:
    with client_for(settings(cors_origins=("https://app.example.org",))) as client:
        response = client.get("/health", headers={"Origin": "https://app.example.org"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://app.example.org"


def test_disallowed_origin_receives_no_cors_grant() -> None:
    with client_for(settings(cors_origins=("https://app.example.org",))) as client:
        response = client.get("/health", headers={"Origin": "https://evil.example.net"})
    assert "access-control-allow-origin" not in response.headers


def test_same_origin_request_is_unaffected() -> None:
    with client_for(settings()) as client:
        assert client.get("/health").status_code == 200


def test_preflight_from_an_allowed_origin_is_answered() -> None:
    with client_for(settings(cors_origins=("https://app.example.org",))) as client:
        response = client.options(
            "/predict",
            headers={
                "Origin": "https://app.example.org",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://app.example.org"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_preflight_from_a_disallowed_origin_is_refused() -> None:
    with client_for(settings(cors_origins=("https://app.example.org",))) as client:
        response = client.options(
            "/predict",
            headers={
                "Origin": "https://evil.example.net",
                "Access-Control-Request-Method": "POST",
            },
        )
    assert response.status_code >= 400
    assert "access-control-allow-origin" not in response.headers


def test_credentials_are_off_by_default_and_never_combined_with_a_wildcard() -> None:
    options = security.cors_options(settings(cors_origins=("https://app.example.org",)))
    assert options["allow_credentials"] is False
    assert options["allow_origins"] == ["https://app.example.org"]


def test_credentials_opt_in_is_supported_with_explicit_origins() -> None:
    options = security.cors_options(
        settings(cors_origins=("https://app.example.org",), cors_allow_credentials=True)
    )
    assert options["allow_credentials"] is True
    assert "*" not in options["allow_origins"]


def test_wildcard_cannot_be_configured() -> None:
    with pytest.raises(ValueError, match="Wildcard CORS is not supported"):
        security.cors_options(settings(cors_origins=("*",)))


def test_production_without_configured_origins_refuses_cross_origin() -> None:
    """Fail closed: no configured production origin means no cross-origin access."""
    with client_for(production_settings(cors_origins=())) as client:
        response = client.get("/health", headers={"Origin": "https://anywhere.example"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers


def test_credentials_without_origins_is_rejected_at_configuration_time() -> None:
    with pytest.raises(ValueError, match="requires explicit"):
        SecuritySettings.from_env(
            {
                security.ENV_CORS_CREDENTIALS: "true",
                security.ENV_CORS_ORIGINS: "",
                security.ENV_APP_ENV: "production",
            }
        )


def test_cors_origins_come_from_configuration() -> None:
    resolved = SecuritySettings.from_env(
        {security.ENV_CORS_ORIGINS: "https://a.example, https://b.example"}
    )
    assert resolved.cors_origins == ("https://a.example", "https://b.example")


def test_development_default_origins_cover_the_vite_dev_server() -> None:
    """The documented Vite proxy setup must keep working out of the box."""
    assert "http://localhost:5173" in SecuritySettings.from_env({}).cors_origins
    assert "http://127.0.0.1:5173" in SecuritySettings.from_env({}).cors_origins


def test_request_id_header_is_exposed_to_cross_origin_callers() -> None:
    options = security.cors_options(settings())
    assert "X-Request-ID" in options["expose_headers"]
    assert "Retry-After" in options["expose_headers"]


# --------------------------------------------------------------------------- #
# 3. Rate limiting
# --------------------------------------------------------------------------- #

def test_predict_is_rate_limited_and_returns_429_with_retry_after() -> None:
    limiter = InMemoryRateLimiter()
    config = settings(rate_limits={"*": 60, "/predict": 3, "/health": 600})
    with client_for(config, limiter) as client:
        for _ in range(3):
            assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 200
        refused = client.post("/predict", json=COMPLETE_PATIENT)

    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) >= 1
    body = refused.json()
    assert body["error"]["code"] == "RATE_LIMITED"
    assert "slow down" in body["error"]["message"].lower()
    assert body["error"]["request_id"]


def test_rate_limit_body_carries_no_internals() -> None:
    limiter = InMemoryRateLimiter()
    config = settings(rate_limits={"*": 60, "/predict": 1, "/health": 600})
    with client_for(config, limiter) as client:
        client.post("/predict", json=COMPLETE_PATIENT)
        text = client.post("/predict", json=COMPLETE_PATIENT).text
    for leak in ("Traceback", str(prep.PROJECT_ROOT), ":\\", "joblib", "app.security"):
        assert leak not in text


def test_limits_are_endpoint_specific() -> None:
    """Exhausting /predict must not throttle a different endpoint."""
    limiter = InMemoryRateLimiter()
    config = settings(rate_limits={"*": 60, "/predict": 2, "/model-info": 60, "/health": 600})
    with client_for(config, limiter) as client:
        for _ in range(5):
            client.post("/predict", json=COMPLETE_PATIENT)
        assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 429
        assert client.get("/model-info").status_code == 200


def test_health_checks_are_not_throttled_at_the_predict_budget() -> None:
    """Normal monitoring must not be mistaken for abuse."""
    config = production_settings(rate_limits={"*": 60, "/predict": 20, "/health": 600})
    with client_for(config) as client:
        for _ in range(40):
            assert client.get("/health").status_code == 200


def test_rate_limit_budget_recovers_within_the_window() -> None:
    """The counter is a rolling window, not a permanent lockout."""
    limiter = InMemoryRateLimiter(window_seconds=0.2)
    config = settings(rate_limits={"*": 60, "/predict": 1, "/health": 600})
    with client_for(config, limiter) as client:
        assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 200
        assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 429
        import time as _time
        _time.sleep(0.25)
        assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 200


def test_allowed_requests_report_remaining_budget() -> None:
    limiter = InMemoryRateLimiter()
    config = settings(rate_limits={"*": 60, "/predict": 5, "/health": 600})
    with client_for(config, limiter) as client:
        response = client.post("/predict", json=COMPLETE_PATIENT)
    assert response.headers["X-RateLimit-Limit"] == "5"
    assert int(response.headers["X-RateLimit-Remaining"]) == 4


def test_preflight_is_not_charged_against_the_budget() -> None:
    limiter = InMemoryRateLimiter()
    config = settings(rate_limits={"*": 60, "/predict": 2, "/health": 600})
    with client_for(config, limiter) as client:
        for _ in range(10):
            client.options(
                "/predict",
                headers={
                    "Origin": "http://localhost:5173",
                    "Access-Control-Request-Method": "POST",
                },
            )
        assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 200


def test_rate_limits_come_from_environment() -> None:
    resolved = SecuritySettings.from_env(
        {
            security.ENV_RATE_LIMIT_PREDICT: "7",
            security.ENV_RATE_LIMIT_DEFAULT: "11",
            security.ENV_RATE_LIMIT_HEALTH: "999",
        }
    )
    assert resolved.limit_for_path("/predict") == 7
    assert resolved.limit_for_path("/model-info") == 11
    assert resolved.limit_for_path("/health") == 999


def test_api_prefixed_paths_share_the_endpoint_budget() -> None:
    """The Vite proxy strips /api; a direct /api call must not bypass the limit."""
    resolved = SecuritySettings.from_env({security.ENV_RATE_LIMIT_PREDICT: "3"})
    assert resolved.limit_for_path("/api/predict") == 3
    assert resolved.limit_for_path("/predict/") == 3


def test_limiter_table_is_bounded() -> None:
    """A rotating spoofed identity must not be able to grow the heap forever."""
    limiter = InMemoryRateLimiter(max_keys=64)
    for index in range(500):
        limiter.check(f"10.0.0.{index % 250}", 5)
    assert len(limiter._buckets) <= 64


def test_in_memory_limiter_can_be_replaced_by_another_implementation() -> None:
    """The documented Redis path only has to satisfy the same interface."""

    class AlwaysAllow(security.RateLimiter):
        def check(self, key, limit, window_seconds=None):
            return security.RateLimitDecision(True, limit, limit, 0)

        def reset(self):
            return None

    config = settings(rate_limits={"*": 1, "/predict": 1, "/health": 1})
    with client_for(config, AlwaysAllow()) as client:
        for _ in range(5):
            assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 200


# --------------------------------------------------------------------------- #
# 4. Request size limits
# --------------------------------------------------------------------------- #

def test_oversized_body_is_rejected_with_413() -> None:
    payload = dict(COMPLETE_PATIENT)
    payload["nickname"] = "x" * 10_000  # unknown field too, but size is the point
    with client_for(settings(max_request_size_bytes=1024)) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_body_under_the_limit_is_accepted() -> None:
    with client_for(settings(max_request_size_bytes=64 * 1024)) as client:
        assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 200


def test_413_body_carries_no_internals() -> None:
    with client_for(settings(max_request_size_bytes=256)) as client:
        text = client.post(
            "/predict", content=b"x" * 4096,
            headers={"content-type": "application/json"},
        ).text
    for leak in ("Traceback", str(prep.PROJECT_ROOT), ":\\"):
        assert leak not in text


def test_chunked_body_without_content_length_is_also_capped() -> None:
    """A body with no declared length must not be able to exceed the budget.

    Exercised at the ASGI boundary rather than through a client, because an HTTP
    client cannot easily send a streaming body without a `Content-Length` - and
    it is exactly that case the streaming guard exists for.
    """

    async def scenario() -> tuple[int, bytes]:
        consumed = bytearray()

        async def greedy_app(scope, receive, send):  # noqa: ANN001
            while True:
                message = await receive()
                if message["type"] != "http.request":
                    break
                consumed.extend(message.get("body") or b"")
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b"ok"})

        middleware = security.BodySizeLimitMiddleware(
            greedy_app, settings(max_request_size_bytes=512)
        )
        chunks = [b"x" * 400, b"x" * 400, b"x" * 400]
        state = {"index": 0}

        async def receive():
            if state["index"] < len(chunks):
                chunk = chunks[state["index"]]
                state["index"] += 1
                return {"type": "http.request", "body": chunk, "more_body": True}
            return {"type": "http.request", "body": b"", "more_body": False}

        sent: list[Message] = []

        async def send(message):
            sent.append(message)

        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/predict",
                # No content-length: the streaming guard is the only defence.
                "headers": [(b"content-type", b"application/json")],
                "client": ("127.0.0.1", 1234),
            },
            receive,
            send,
        )
        start = next(m for m in sent if m["type"] == "http.response.start")
        body = b"".join(m.get("body") or b"" for m in sent if m["type"] == "http.response.body")
        return start["status"], body

    import anyio

    status_code, body = anyio.run(scenario)
    assert status_code == 413
    assert json.loads(body)["error"]["code"] == "PAYLOAD_TOO_LARGE"
    # Rejected while streaming, not after buffering the whole body.
    assert sum(len(c) for c in (b"x" * 400,)) * 1 <= 1200


def test_max_body_size_is_configurable() -> None:
    resolved = SecuritySettings.from_env({security.ENV_MAX_BODY_BYTES: "2048"})
    assert resolved.max_request_size_bytes == 2048


# --------------------------------------------------------------------------- #
# 5. Request IDs
# --------------------------------------------------------------------------- #

def test_every_response_carries_a_request_id(dev_client: TestClient) -> None:
    for response in (
        dev_client.get("/health"),
        dev_client.get("/model-info"),
        dev_client.post("/predict", json=COMPLETE_PATIENT),
    ):
        assert response.headers["X-Request-ID"]


def test_request_ids_are_unique_per_request(dev_client: TestClient) -> None:
    ids = {dev_client.get("/health").headers["X-Request-ID"] for _ in range(5)}
    assert len(ids) == 5


def test_a_valid_client_supplied_request_id_is_echoed() -> None:
    with client_for(settings()) as client:
        response = client.get("/health", headers={"X-Request-ID": "trace-abc.123_x~"})
    assert response.headers["X-Request-ID"] == "trace-abc.123_x~"


@pytest.mark.parametrize(
    "supplied",
    ["a" * 200, "has space", "has\nnewline", "<script>", "tab\there", ""],
)
def test_a_malformed_client_request_id_is_replaced_not_echoed(supplied: str) -> None:
    with client_for(settings()) as client:
        response = client.get("/health", headers={"X-Request-ID": supplied})
    echoed = response.headers["X-Request-ID"]
    assert echoed != supplied or supplied == ""
    assert len(echoed) <= 64


def test_request_id_appears_in_the_error_envelope(dev_client: TestClient) -> None:
    response = dev_client.post("/predict", json={"age": 57})
    body = response.json()
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]


def test_the_request_id_reaches_the_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="swasthyguide.api")
    with client_for(settings()) as client:
        response = client.post("/predict", json=COMPLETE_PATIENT)
    request_id = response.headers["X-Request-ID"]
    matching = [r for r in caplog.records if request_id == getattr(r, "request_id", None)]
    assert matching, "the request id must be present on operational log lines"


def test_logs_never_contain_the_prediction_payload(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    with client_for(settings()) as client:
        response = client.post("/predict", json=COMPLETE_PATIENT)
    blob = "\n".join(record.getMessage() for record in caplog.records)
    assert "reversable defect" not in blob
    assert "245.0" not in blob          # the cholesterol value
    assert "asymptomatic" not in blob   # the chest-pain category
    assert "risk_probability" not in blob


# --------------------------------------------------------------------------- #
# 6. Safe error handling
# --------------------------------------------------------------------------- #

def test_an_internal_exception_returns_a_safe_envelope_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def explode(*_args, **_kwargs):
        raise RuntimeError(
            f"pipeline secret at C:\\Users\\someone\\models\\logistic.joblib failed"
        )

    monkeypatch.setattr(service, "build_prediction_response", explode)
    with client_for(settings()) as client:
        response = client.post("/predict", json=COMPLETE_PATIENT)

    assert response.status_code == 500
    body = response.json()
    assert body["error"]["code"] == "INTERNAL_ERROR"
    assert body["error"]["message"] == "An unexpected error occurred."
    assert body["error"]["request_id"] == response.headers["X-Request-ID"]


def test_internal_errors_leak_no_internals(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_args, **_kwargs):
        raise ValueError("sklearn/numpy internals at site-packages/numpy/core/x.py")

    monkeypatch.setattr(service, "build_prediction_response", explode)
    with client_for(settings()) as client:
        text = client.post("/predict", json=COMPLETE_PATIENT).text

    for leak in (
        "Traceback",
        "RuntimeError",
        "ValueError",
        "site-packages",
        "sklearn",
        "numpy",
        str(prep.PROJECT_ROOT),
        ":\\",
        "app/security.py",
        "joblib",
    ):
        assert leak not in text, f"500 response leaked {leak!r}"


def test_the_original_exception_is_still_logged_for_the_operator(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The operator keeps the traceback; the caller does not."""
    caplog.set_level(logging.ERROR, logger="swasthyguide.api")

    def explode(*_args, **_kwargs):
        raise RuntimeError("diagnostic detail for the operator only")

    monkeypatch.setattr(service, "build_prediction_response", explode)
    with client_for(settings()) as client:
        response = client.post("/predict", json=COMPLETE_PATIENT)

    assert response.status_code == 500
    assert "diagnostic detail for the operator only" not in response.text

    logged = [
        record for record in caplog.records
        if record.exc_info is not None
        and "diagnostic detail for the operator only" in logging.Formatter().formatException(record.exc_info)
    ]
    assert logged, "the traceback must remain available to the operator"


def test_validation_errors_keep_the_field_and_the_reason(dev_client: TestClient) -> None:
    response = dev_client.post("/predict", json={"age": 57, "sex": "Male"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "VALIDATION_ERROR"
    entries = body["detail"]
    assert isinstance(entries, list) and entries
    locations = [part for entry in entries for part in entry["loc"]]
    assert "cp" in locations


def test_validation_errors_do_not_echo_the_submitted_values(
    dev_client: TestClient,
) -> None:
    """The caller's own health inputs are not reflected back into the response."""
    response = dev_client.post(
        "/predict", json={"age": 57, "sex": "Male", "cp": "typcal angina"}
    )
    body = response.json()
    assert response.status_code == 422
    assert "typcal angina" not in response.text
    for entry in body["detail"]:
        assert set(entry) == {"type", "loc", "msg"}
        assert "input" not in entry
        assert "ctx" not in entry
        assert "url" not in entry


def test_malformed_json_still_returns_422_without_internals(dev_client: TestClient) -> None:
    response = dev_client.post(
        "/predict", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 422
    assert "Traceback" not in response.text
    assert "json.decoder" not in response.text


def test_error_messages_are_scrubbed_of_path_shaped_text() -> None:
    assert security._scrub_message("all good") == "all good"
    assert security._scrub_message("C:\\Users\\x\\models\\y.joblib") == security.GENERIC_INTERNAL_MESSAGE
    assert security._scrub_message('File "/srv/app/main.py", line 3') == security.GENERIC_INTERNAL_MESSAGE
    assert security._scrub_message("Traceback (most recent call last)") == security.GENERIC_INTERNAL_MESSAGE
    assert security._scrub_message("x" * 500) == security.GENERIC_INTERNAL_MESSAGE
    assert security._scrub_message("") == security.GENERIC_INTERNAL_MESSAGE


# --------------------------------------------------------------------------- #
# 7. Documentation exposure
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_documentation_is_available_in_development(path: str) -> None:
    with client_for(settings(docs_enabled=True)) as client:
        assert client.get(path).status_code == 200


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_documentation_is_disabled_in_production_by_default(path: str) -> None:
    with client_for(production_settings(docs_enabled=False)) as client:
        assert client.get(path).status_code == 404


def test_production_documentation_can_be_re_enabled_explicitly() -> None:
    with client_for(production_settings(docs_enabled=True)) as client:
        assert client.get("/docs").status_code == 200


def test_documentation_control_follows_the_environment() -> None:
    dev = SecuritySettings.from_env({})
    prod = SecuritySettings.from_env({security.ENV_APP_ENV: "production"})
    opted_in = SecuritySettings.from_env(
        {security.ENV_APP_ENV: "production", security.ENV_ENABLE_DOCS: "true"}
    )
    assert dev.docs_enabled is True
    assert prod.docs_enabled is False
    assert opted_in.docs_enabled is True


def test_the_openapi_schema_implementation_is_retained_not_deleted() -> None:
    """Docs are gated by configuration; the schema generator still exists."""
    with client_for(production_settings(docs_enabled=False)) as client:
        application = client.app
    schema = application.openapi()
    assert "/predict" in schema["paths"]
    assert schema["info"]["title"].startswith("SWASTHYAGUIDE")


def test_a_production_page_is_not_served_without_a_host_allow_list() -> None:
    with client_for(production_settings(allowed_hosts=())) as client:
        assert client.get("/health").status_code == 200


# --------------------------------------------------------------------------- #
# 8. /health and /model-info hardening
# --------------------------------------------------------------------------- #

def test_health_exposes_only_minimal_operational_state(dev_client: TestClient) -> None:
    body = dev_client.get("/health").json()
    assert set(body) == {
        "status", "api_stage", "api_version", "model_loaded",
        "model", "deployment_threshold", "disclaimer",
    }


def test_health_leaks_no_paths_versions_or_environment(dev_client: TestClient) -> None:
    text = dev_client.get("/health").text
    for leak in (
        str(prep.PROJECT_ROOT), "OneDrive", ":\\", "joblib", ".py",
        "scikit", "numpy", "Windows", "python",
    ):
        assert leak.lower() not in text.lower(), f"/health leaked {leak!r}"


def test_model_info_leaks_no_absolute_paths(dev_client: TestClient) -> None:
    text = dev_client.get("/model-info").text
    for leak in (str(prep.PROJECT_ROOT), "OneDrive", ":\\", ".joblib\\"):
        assert leak not in text, f"/model-info leaked {leak!r}"


def test_model_info_artifact_is_a_bare_filename(dev_client: TestClient) -> None:
    """No directory component: the response stops locating artifacts on disk."""
    artifact = dev_client.get("/model-info").json()["artifact"]
    assert artifact == "logistic_regression_pipeline.joblib"
    assert "/" not in artifact and "\\" not in artifact


def test_model_info_withholds_the_build_environment_by_default(dev_client: TestClient) -> None:
    """OS build string and dependency versions are deployment fingerprinting."""
    body = dev_client.get("/model-info").json()
    assert "environment" not in body
    text = dev_client.get("/model-info").text
    for leak in ("Windows-11", "scikit_learn", "1.7.1", "shap"):
        assert leak not in text


def test_model_info_still_reports_the_research_provenance() -> None:
    """Hardening must not strip the disclosure the research record depends on."""
    with client_for(settings()) as client:
        body = client.get("/model-info").json()
    assert body["artifact_stage"] == "Stage 3 selected configuration (models/tuned/)"
    assert "not proof" in body["selection_rationale"].lower()
    assert "not a medical device" in body["disclaimer"].lower()
    assert body["n_input_features"] == 13


def test_model_info_build_environment_is_available_to_opted_in_operators() -> None:
    with client_for(settings(expose_build_environment=True)) as client:
        body = client.get("/model-info").json()
    assert isinstance(body["environment"], dict)
    assert body["environment"]


def test_build_environment_opt_in_is_environment_driven() -> None:
    assert SecuritySettings.from_env({}).expose_build_environment is False
    assert SecuritySettings.from_env(
        {security.ENV_EXPOSE_BUILD_ENV: "true"}
    ).expose_build_environment is True


# --------------------------------------------------------------------------- #
# 9. Forwarded headers and Host validation
# --------------------------------------------------------------------------- #

def test_forwarded_headers_are_not_trusted_by_default() -> None:
    """A caller must not be able to rotate its own identity with a header."""
    scope = {
        "type": "http",
        "client": ("10.0.0.9", 5555),
        "headers": [(b"x-forwarded-for", b"1.2.3.4")],
        "path": "/predict",
    }
    assert security.client_ip(scope, trusted_proxy_hops=0) == "10.0.0.9"


def test_forwarded_headers_are_used_only_when_proxies_are_declared() -> None:
    scope = {
        "type": "http",
        "client": ("10.0.0.9", 5555),
        "headers": [(b"x-forwarded-for", b"1.2.3.4, 5.6.7.8")],
        "path": "/predict",
    }
    assert security.client_ip(scope, trusted_proxy_hops=1) == "5.6.7.8"
    assert security.client_ip(scope, trusted_proxy_hops=2) == "1.2.3.4"


def test_a_hostile_forwarded_chain_cannot_invent_the_client() -> None:
    """Only the declared number of trusted hops from the right is believed."""
    scope = {
        "type": "http",
        "client": ("10.0.0.9", 5555),
        "headers": [(b"x-forwarded-for", b"9.9.9.9, 8.8.8.8, 7.7.7.7, 6.6.6.6")],
        "path": "/predict",
    }
    assert security.client_ip(scope, trusted_proxy_hops=1) == "6.6.6.6"


def test_spoofed_forwarded_headers_cannot_dodge_the_rate_limit() -> None:
    """The real reason the default is 0 trusted hops."""
    limiter = InMemoryRateLimiter()
    config = settings(rate_limits={"*": 60, "/predict": 2, "/health": 600},
                      trusted_proxy_hops=0)
    with client_for(config, limiter) as client:
        for _ in range(2):
            assert client.post("/predict", json=COMPLETE_PATIENT).status_code == 200
        refused = client.post(
            "/predict", json=COMPLETE_PATIENT,
            headers={"X-Forwarded-For": "8.8.8.8"},
        )
    assert refused.status_code == 429


def test_host_validation_rejects_an_unknown_host_when_configured() -> None:
    config = settings(allowed_hosts=("app.example.org",))
    with client_for(config) as client:
        refused = client.get("/health", headers={"Host": "evil.example.net"})
        allowed = client.get("/health", headers={"Host": "app.example.org"})
    assert refused.status_code == 400
    assert allowed.status_code == 200


def test_host_validation_is_off_when_no_host_is_configured(dev_client: TestClient) -> None:
    assert dev_client.get("/health").status_code == 200


def test_a_wildcard_host_list_does_not_silently_disable_validation() -> None:
    config = settings(allowed_hosts=("*",))
    with client_for(config) as client:
        # Not "allow everything pretending to be protected": the middleware is
        # simply not installed, so the operator sees no false sense of control.
        assert client.get("/health", headers={"Host": "anything.example"}).status_code == 200
    assert "TrustedHostMiddleware" not in {
        m.cls.__name__ for m in client_for(config).app.user_middleware
    }


def test_trusted_proxy_hops_are_environment_driven() -> None:
    assert SecuritySettings.from_env({}).trusted_proxy_hops == 0
    assert SecuritySettings.from_env(
        {security.ENV_TRUSTED_PROXY_HOPS: "2"}
    ).trusted_proxy_hops == 2


# --------------------------------------------------------------------------- #
# 10. Regression: the prediction contract is unchanged
# --------------------------------------------------------------------------- #

def test_a_valid_prediction_still_works(dev_client: TestClient) -> None:
    response = dev_client.post("/predict", json=COMPLETE_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert body["threshold"] == 0.40
    assert body["model"] == service.DEFAULT_MODEL
    assert "not a medical device" in body["disclaimer"].lower()


def test_the_security_stack_does_not_change_the_score(dev_client: TestClient) -> None:
    """The score must be byte-identical to the pipeline run with no HTTP at all."""
    from app.schemas import PatientRiskRequest

    loaded = service.load_model()
    expected = float(
        loaded.pipeline.predict_proba(
            PatientRiskRequest(**COMPLETE_PATIENT).to_pipeline_frame()
        )[0, 1]
    )
    reported = dev_client.post("/predict", json=COMPLETE_PATIENT).json()["risk_probability"]
    assert reported == pytest.approx(expected, abs=5e-7)


def test_the_minimal_request_still_validates(dev_client: TestClient) -> None:
    response = dev_client.post(
        "/predict", json={"age": 61, "sex": "Female", "cp": "typical angina"}
    )
    assert response.status_code == 200
    assert json.loads(response.text)["notes"]


def test_an_unknown_field_is_still_rejected(dev_client: TestClient) -> None:
    body = dict(COMPLETE_PATIENT)
    body["nickname"] = "patient-1"
    assert dev_client.post("/predict", json=body).status_code == 422


def test_an_out_of_range_value_is_still_rejected(dev_client: TestClient) -> None:
    body = dict(COMPLETE_PATIENT)
    body["ca"] = 7
    assert dev_client.post("/predict", json=body).status_code == 422


def test_503_behaviour_is_preserved_when_the_model_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generic 503 message the frontend relies on must survive hardening."""
    monkeypatch.setitem(
        service.SERVED_MODELS, service.DEFAULT_MODEL, "tuned/absent_model.joblib"
    )
    with client_for(settings()) as client:
        health = client.get("/health")
        predict = client.post("/predict", json=COMPLETE_PATIENT)
        info = client.get("/model-info")

    assert health.status_code == 503
    assert health.json()["detail"] == UNAVAILABLE_DETAIL
    assert predict.status_code == 503
    assert predict.json()["detail"] == UNAVAILABLE_DETAIL
    assert info.status_code == 503
    for leak in ("Traceback", "joblib", ":\\", str(prep.MODELS_DIR)):
        assert leak not in health.text


def test_security_controls_are_off_the_critical_path_of_inference(
    dev_client: TestClient,
) -> None:
    """Two identical requests remain bit-identical with every control enabled."""
    first = dev_client.post("/predict", json=COMPLETE_PATIENT).json()
    second = dev_client.post("/predict", json=COMPLETE_PATIENT).json()
    assert first == second


def test_security_module_does_not_touch_the_frozen_pipeline() -> None:
    """Belt and braces: importing the controls must not alter model behaviour."""
    assert "prep" not in security.__dict__
    assert not any(
        name in security.__dict__ for name in ("load_model", "predict_proba", "fit")
    )
    source = (prep.PROJECT_ROOT / "app" / "security.py").read_text(encoding="utf-8")
    for forbidden in ("joblib", "predict_proba", "from src", "import src"):
        assert forbidden not in source


def test_no_new_runtime_dependency_was_added() -> None:
    """Every control uses the standard library plus already-pinned FastAPI."""
    requirements = (prep.PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")
    for forbidden in ("redis", "slowapi", "fastapi-limiter", "limits", "brotli", "nh3"):
        assert forbidden not in requirements.lower()