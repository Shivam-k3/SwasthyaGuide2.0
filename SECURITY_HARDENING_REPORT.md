# SECURITY_HARDENING_REPORT.md
**Repository:** Swasthyaguide2.0
**Date:** 2026-10-02 (Stage 1B: runtime implementation)
**Stage 1 (audit + documentation):** complete, retained
**Stage 1B (runtime enforcement):** complete

## A. Executive Summary

Stage 1 produced the audit, threat model, OWASP assessment and control designs but
enforced **zero** executable controls: everything was documented and nothing was
running.

Stage 1B implements those controls at the application boundary. Seven controls are
now runtime-enforced with 91 new tests covering them, plus
`app/security.py` and `tests/test_security.py` as new files and `app/main.py`,
`app/model_service.py` and `app/schemas.py` modified at the API boundary only.

No new runtime dependency was added. Every control uses the standard library plus
the FastAPI/Starlette middleware already pinned in `requirements.txt`.

The frozen ML pipeline is untouched. `data/`, `src/`, `models/`, `notebooks/` and
`STAGE_3_AUDIT.md` are byte-identical across all 81 source files, the dataset
SHA-256 is unchanged, and `python -m src.verify_stage3` reports 69/69. Prediction
behaviour is unchanged: a test asserts the HTTP score matches a direct pipeline
call to within 5e-7.

## B. Pre-Hardening State

Recorded in `SECURITY_AUDIT_PRE.md` (retained as the historical baseline).

- Backend: 123 tests passing
- Frontend: 49 tests passing; TypeScript clean; production build succeeds
- Security: no security headers, no rate limiting, no CORS configuration, docs
  always exposed, no centralised safe error handling, no request IDs
- A half-written `app/security.py` existed but was unwired, unused, and contained
  an unconditional trust of `X-Forwarded-For`
- `app/main.py` carried a dead-code expression, `None if False and os.getenv(...)`,
  which evaluated to "docs always on" and would have raised `NameError` had the
  `False` ever been removed

## C. Threat Model

Retained in `SECURITY_THREAT_MODEL.md`. The two threats it identified as primary
were automated abuse of `/predict` (API4) and security misconfiguration (API8).
Both are now mitigated in code; see E and the residual-risk table in G.

## D. OWASP API Security Top 10 Assessment

Retained in `OWASP_API_SECURITY_ASSESSMENT.md`. API4 and API8 applied; the
remaining categories remain not-applicable for a stateless anonymous API with no
object identifiers, no external API consumption, and no outbound network calls.

API4 (Unrestricted Resource Consumption) and API8 (Security Misconfiguration) moved
from "mitigation designed" to "mitigation enforced and tested".

## E. Controls implemented in code

Every row below is enforced at runtime and has at least one test in
`tests/test_security.py`. Test names are given so each claim is checkable.

### E1. Security headers — `app/security.py`, `SecurityHeadersMiddleware`

`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`,
`Referrer-Policy: no-referrer`, `Permissions-Policy` (camera/geolocation/mic and
others denied), `Content-Security-Policy`, `Cross-Origin-Opener-Policy`,
`Cross-Origin-Resource-Policy`, `X-DNS-Prefetch-Control: off`,
`Cache-Control: no-store`, `Pragma: no-cache`.

The CSP is `default-src 'none'` with explicit allowances for the assets FastAPI's
own Swagger UI and ReDoc pages load (`cdn.jsdelivr.net` plus inline bootstrap).
It was written against the shipped documentation pages, not copied from a generic
template; a test asserts the Swagger asset sources remain reachable so the control
cannot silently break our own docs.

HSTS is emitted only when `SWASTHYAGUIDE_FORCE_HTTPS=true` or
`SWASTHYAGUIDE_ENV=production`. It is absent in local HTTP development on purpose:
pinning `localhost` to HTTPS in a developer's browser would break their own setup.

Tests: `test_security_headers_are_present_on_a_normal_response`,
`test_cache_control_prevents_intermediary_caching`,
`test_hsts_is_absent_during_local_http_development`,
`test_hsts_is_present_when_forced_https`,
`test_hsts_is_present_in_production_by_default`,
`test_headers_are_present_on_error_responses_too`,
`test_csp_is_configurable`, `test_csp_allows_the_shipped_swagger_ui_assets`.

### E2. CORS — `app/security.py`, `SecuritySettings.from_env` + `cors_options`

Explicit origin allow-list from `CORS_ALLOWED_ORIGINS`. Development defaults to the
Vite dev server and the API's own origins; production defaults to an **empty**
list, which refuses every cross-origin request. Fail-closed, not fail-open.

There is no code path that yields `*`: `cors_options` raises on a wildcard origin,
and a test asserts that. `CORS_ALLOW_CREDENTIALS` defaults off and is rejected at
configuration time if enabled together with an empty allow-list, so credentials
can never be combined with a wildcard.

The existing Vite `/api` proxy is untouched and still works: same-origin calls
never depend on CORS. The development origin defaults keep a direct cross-origin
call working too.

Tests: `test_allowed_origin_receives_cors_headers`,
`test_disallowed_origin_receives_no_cors_grant`,
`test_preflight_from_an_allowed_origin_is_answered`,
`test_preflight_from_a_disallowed_origin_is_refused`,
`test_credentials_are_off_by_default_and_never_combined_with_a_wildcard`,
`test_credentials_opt_in_is_supported_with_explicit_origins`,
`test_wildcard_cannot_be_configured`,
`test_production_without_configured_origins_refuses_cross_origin`,
`test_credentials_without_origins_is_rejected_at_configuration_time`,
`test_cors_origins_come_from_configuration`,
`test_development_default_origins_cover_the_vite_dev_server`,
`test_request_id_header_is_exposed_to_cross_origin_callers`.

### E3. Rate limiting — `app/security.py`, `RateLimitMiddleware` + `InMemoryRateLimiter`

Real enforcement, not documentation. `/predict` defaults to 20 requests per
minute per client; `/health` defaults to 600 so a monitoring agent polling every
few seconds is never mistaken for abuse; everything else defaults to 60. Exceeding
a budget returns **429** with a **`Retry-After`** header and a `RATE_LIMITED` error
body. Allowed requests carry `X-RateLimit-Limit` and `X-RateLimit-Remaining`.

`OPTIONS` preflight is exempt, because a browser sends one before every
cross-origin POST and counting it would make the limit depend on browser
behaviour rather than caller behaviour.

The limiter sits behind a two-method interface (`check`, `reset`) with no framework
types, so a Redis implementation is a drop-in replacement with no call-site
changes. Redis is **not** a dependency and does not need to be until the
deployment is multi-instance.

**IN-MEMORY RATE LIMITING IS NOT SUFFICIENT FOR MULTI-INSTANCE PRODUCTION.**
State is per-process: N instances behind a load balancer give a caller N times the
configured limit. Move `security.RateLimiter` to a shared store before scaling out.

The counter table is bounded, so rotating spoofed client addresses cannot grow the
process heap without limit.

Tests: `test_predict_is_rate_limited_and_returns_429_with_retry_after`,
`test_rate_limit_body_carries_no_internals`, `test_limits_are_endpoint_specific`,
`test_health_checks_are_not_throttled_at_the_predict_budget`,
`test_rate_limit_budget_recovers_within_the_window`,
`test_allowed_requests_report_remaining_budget`,
`test_preflight_is_not_charged_against_the_budget`,
`test_rate_limits_come_from_environment`,
`test_api_prefixed_paths_share_the_endpoint_budget`,
`test_limiter_table_is_bounded`,
`test_in_memory_limiter_can_be_replaced_by_another_implementation`.

### E4. Request size limits — `app/security.py`, `BodySizeLimitMiddleware`

`Content-Length` is checked first, which settles the ordinary case without reading
a byte. A body with no declared length is caught by wrapping `receive` and failing
the request as soon as the budget is passed, so an undeclared large body cannot be
streamed in under the limit. Oversized requests get **413** with a
`PAYLOAD_TOO_LARGE` body, before any prediction work is attempted.

Default 1 MiB via `MAX_REQUEST_SIZE_BYTES`. Frontend validation is not relied on.
Image-upload limits are deliberately not implemented: the application has no
upload path.

Tests: `test_oversized_body_is_rejected_with_413`,
`test_body_under_the_limit_is_accepted`, `test_413_body_carries_no_internals`,
`test_chunked_body_without_content_length_is_also_capped`,
`test_max_body_size_is_configurable`.

### E5. Request IDs — `app/security.py`, `RequestIdMiddleware` + `RequestIdFilter`

Every request receives an ID and every response returns it in `X-Request-ID`,
including 413, 429, 422 and 500 responses.

A client-supplied ID is accepted only if it matches `[A-Za-z0-9._~-]{1,64}`.
Anything else - over-long, containing whitespace, newlines, tabs or markup - is
replaced with a fresh UUID and the rejection is logged. This blocks log forging
through a correlation header.

The ID reaches the logs through a `ContextVar` and a logging filter, so it is
present on operational lines without any call site having to pass it. The access
log records method, path, status, duration and client address. The query string is
dropped. **No request body, health input, or prediction result is ever passed to a
logger**, and a test asserts the submitted cholesterol value, chest-pain category
and response field names are absent from the captured log text.

Tests: `test_every_response_carries_a_request_id`,
`test_request_ids_are_unique_per_request`,
`test_a_valid_client_supplied_request_id_is_echoed`,
`test_a_malformed_client_request_id_is_replaced_not_echoed` (6 cases),
`test_request_id_appears_in_the_error_envelope`,
`test_the_request_id_reaches_the_logs`,
`test_logs_never_contain_the_prediction_payload`.

### E6. Centralised safe error handling — `app/security.py`, `SafeErrorMiddleware` + handlers

Unhandled exceptions become:

```json
{
  "detail": "An unexpected error occurred.",
  "error": {
    "code": "INTERNAL_ERROR",
    "message": "An unexpected error occurred.",
    "request_id": "..."
  }
}
```

`SafeErrorMiddleware` sits inside Starlette's error middleware, handles the
exception and does not re-raise, so the client gets the safe envelope while the
operator keeps the full traceback in the log. Every message is passed through
`_scrub_message`, which rejects anything carrying a traceback header, a source-file
reference, a Windows or UNC path, or a Python package directory, and clamps length.

Intended behaviour is preserved:

- **422** keeps `detail` as a list of `{type, loc, msg}`, which is what the frontend
  maps onto form fields. FastAPI's default `input`, `ctx` and `url` keys are
  dropped: `input` echoes the caller's own health values back into the response and
  into any intermediary log, and `ctx` can embed exception objects.
- **503** passes the existing constant `UNAVAILABLE_DETAIL` through byte-identical,
  because the frontend already depends on that exact message.
- **429** and **413** use the same envelope with their own codes.

Tests: `test_an_internal_exception_returns_a_safe_envelope_not_a_traceback`,
`test_internal_errors_leak_no_internals` (9 leak patterns),
`test_the_original_exception_is_still_logged_for_the_operator`,
`test_validation_errors_keep_the_field_and_the_reason`,
`test_validation_errors_do_not_echo_the_submitted_values`,
`test_malformed_json_still_returns_422_without_internals`,
`test_error_messages_are_scrubbed_of_path_shaped_text`.

### E7. Production API documentation control — `app/main.py`, `create_app`

`/docs`, `/redoc` and `/openapi.json` are available in development and return 404
in production by default. `SWASTHYAGUIDE_ENABLE_DOCS=true` re-enables them
explicitly. The documentation implementation is **not** deleted: `app.openapi()`
still serves the full schema in-process, and a test asserts `/predict` is present
in it.

Tests: `test_documentation_is_available_in_development` (3 paths),
`test_documentation_is_disabled_in_production_by_default` (3 paths),
`test_production_documentation_can_be_re_enabled_explicitly`,
`test_documentation_control_follows_the_environment`,
`test_the_openapi_schema_implementation_is_retained_not_deleted`.

### E8. Host and forwarded-header safety — `app/security.py`

`client_ip()` ignores `X-Forwarded-For` and `X-Real-IP` entirely unless
`SWASTHYAGUIDE_TRUSTED_PROXY_HOPS` declares how many proxies are in front. It then
reads the Nth entry from the **right**, which is the only position a proxy we
control actually appended; everything to its left is attacker-supplied and
disregarded. This replaces the Stage 1 stub, which trusted `X-Forwarded-For`
unconditionally and would have let a caller rotate its identity to evade any limit.

`TrustedHostMiddleware` is installed only when `ALLOWED_HOSTS` is set and does not
contain `*`. A wildcard list does not install the middleware and does not pretend
to: there is no false sense of control. When no host list is configured the
application logs a warning that Host validation is disabled, rather than leaving it
to be assumed.

The proxy must strip client-supplied `X-Forwarded-For` before setting its own; this
is stated in `PRODUCTION_SECURITY_CHECKLIST.md` because the application cannot
enforce it.

Tests: `test_forwarded_headers_are_not_trusted_by_default`,
`test_forwarded_headers_are_used_only_when_proxies_are_declared`,
`test_a_hostile_forwarded_chain_cannot_invent_the_client`,
`test_spoofed_forwarded_headers_cannot_dodge_the_rate_limit`,
`test_host_validation_rejects_an_unknown_host_when_configured`,
`test_host_validation_is_off_when_no_host_is_configured`,
`test_a_wildcard_host_list_does_not_silently_disable_validation`,
`test_trusted_proxy_hops_are_environment_driven`.

### E9. `/health` hardening — `app/main.py`

Unchanged shape, verified minimal. The response is exactly `status`,
`api_stage`, `api_version`, `model_loaded`, `model`, `deployment_threshold`,
`disclaimer`. No filesystem path, no model location, no dependency inventory, no
environment variable. `deployment_threshold` is retained deliberately: it is
already public via `/model-info`, so withholding it here would reduce monitoring
signal without reducing disclosure.

Tests: `test_health_exposes_only_minimal_operational_state`,
`test_health_leaks_no_paths_versions_or_environment`.

### E10. `/model-info` hardening — `app/model_service.py`, `app/schemas.py`

Two changes, both reducing disclosure:

1. `artifact` is now a **bare filename** (`logistic_regression_pipeline.joblib`)
   rather than a repository-relative path. The location was already stated in
   prose in `artifact_stage` and `threshold_source`, so no provenance is lost, and
   a test confirms the field contains no directory separator.
2. The Stage 3 `environment` block is withheld unless
   `SWASTHYAGUIDE_EXPOSE_BUILD_ENV=true`. It contains an OS build string and exact
   dependency versions (Python, scikit-learn, numpy, pandas, shap) - a targeting aid
   for known CVEs that tells a caller nothing about the risk assessment. The block
   still exists in the frozen report; only the republication is gated.

The frontend does not consume `/model-info` (verified by search), so no existing
frontend functionality depends on either field.

Tests: `test_model_info_leaks_no_absolute_paths`,
`test_model_info_artifact_is_a_bare_filename`,
`test_model_info_withholds_the_build_environment_by_default`,
`test_model_info_still_reports_the_research_provenance`,
`test_model_info_build_environment_is_available_to_opted_in_operators`,
`test_build_environment_opt_in_is_environment_driven`.

### E11. Startup configuration logging — `app/security.py`, `log_configuration`

A control that is not configured is stated in the startup log rather than left to
be assumed: effective rate limits, body budget, docs and HSTS state, whether Host
validation is active, and whether forwarded headers are trusted.

## F. Dependency and dependency-scanning review

No package was added. Every control uses the standard library plus FastAPI /
Starlette middleware already pinned in `requirements.txt`. `requirements.txt` is
unchanged, which matters beyond tidiness: `src/verify_stage3.py` asserts that no
entry is unpinned, and its 69/69 result depends on that file's integrity.

`pip-audit` and `npm audit` are available but were not run as part of this stage;
they belong in CI. A test asserts no rate-limiting or security-framework package
was introduced.

Test: `test_no_new_runtime_dependency_was_added`.

## G. Regression results

| Check | Result |
|---|---|
| Backend `python -m pytest -q` | **214 passed** (123 pre-existing + 91 new) |
| Security suite `python -m pytest tests/test_security.py -q` | **91 passed** |
| Stage 3 `python -m src.verify_stage3` | **69/69 checks passed** |
| Frontend `npm test` | **49 passed** (4 files) |
| TypeScript `npx tsc -b --noEmit` | exit **0** |
| Production build `npm run build` | success |
| Live API checks (uvicorn, bounded, then terminated) | see H |
| Frozen artifact checksums (81 source files) | **identical** |
| Dataset SHA-256 | `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae` - unchanged |

Prediction behaviour is unchanged. `test_the_security_stack_does_not_change_the_score`
(in `tests/test_security.py`) asserts the HTTP response matches a direct
`predict_proba` call on the persisted pipeline within 5e-7 with the full stack
active. The pre-existing determinism tests still pass, including
`test_a_patient_is_unaffected_by_other_patients` and the order-independence tests in
`tests/test_api.py`.

## H. Live verification performed

Uvicorn was started as a background process with a known PID, checked, and
explicitly terminated. Port 8000 was confirmed free afterwards.

| Check | Result |
|---|---|
| `/`, `/assessment`, `/how-it-works`, `/about` via the Vite dev server | HTTP 200, SPA mount present on all four |
| Security headers through the `/api` proxy (the browser's real path) | all present; `Strict-Transport-Security` correctly absent on local HTTP |
| Valid prediction through the proxy | HTTP 200, `risk_probability` 0.968495, `elevated_risk`, threshold 0.4, disclaimer present |
| 422 through the proxy | HTTP 422, `code=VALIDATION_ERROR`, field `body.cp` identified, `request_id` present, submitted values not echoed, no `input` key |
| Rate limit, 25 sequential `/predict` calls | calls 1-20 = 200, calls 21-25 = **429** with `Retry-After: 58`, `code=RATE_LIMITED`, no internals leaked |
| Body size limit, 1.5 MB body | **HTTP 413**, `code=PAYLOAD_TOO_LARGE` |
| `/health` after the `/predict` budget was exhausted | HTTP 200 - budgets are endpoint-specific |
| CORS, allowed vs disallowed origin | `http://localhost:5173` granted; `https://evil.example.net` not granted |
| `X-Request-ID` supplied as `sw-verify-001` | echoed back unchanged |
| `X-Request-ID` supplied as `bad id with spaces` | replaced with a generated UUID |
| `/model-info` `artifact` | `logistic_regression_pipeline.joblib`, no directory separator |
| `/model-info` `environment` | absent (withheld by default) |
| Leak scan across live responses | no traceback, no package path, no user directory, no OS build string, no dependency versions |

**Not verified in this environment:** there is no browser automation available
(confirmed absent: Playwright and Puppeteer are not installed). Visual rendering,
devtools console errors, horizontal overflow and assistive-technology behaviour
were therefore **not** checked in a live browser. The 49 jsdom tests and the
successful production build cover component rendering and layout logic, but they are
not a substitute for a real browser pass. This is stated as a gap rather than
claimed as verified.

## I. Frozen ML verification

- `data/` unchanged
- `src/` unchanged
- `models/` unchanged
- `notebooks/` unchanged
- `STAGE_3_AUDIT.md` unchanged
- All 81 source files across those paths: SHA-256 identical to the pre-Stage-1B
  baseline. (A first pass also counted 8 generated `src/__pycache__/*.pyc` files
  for 89; `.pyc` output is excluded because re-running pytest legitimately
  rewrites it. No source file differs.)
- Dataset SHA-256: `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae`
- `python -m src.verify_stage3`: 69/69
- Prediction semantics: unchanged, asserted to 5e-7 against the persisted pipeline

`tests/test_security.py` includes a structural guard asserting `app/security.py`
contains no reference to `joblib`, `predict_proba`, or `src`, so the security layer
cannot quietly acquire a dependency on the frozen pipeline.

## J. Remaining risks

1. **In-memory rate limiting is per process.** Correct for the single-instance
   deployment this prototype targets; wrong the moment it is scaled out. Highest
   residual risk in this stage.
2. **Host validation is off until `ALLOWED_HOSTS` is set.** Chosen over a default
   that would either allow everything silently or break a deployment whose hostname
   the application cannot know. Mitigated by a startup warning; still an operator
   obligation.
3. **Forwarded-header correctness depends on the reverse proxy.** The application
   ignores forwarded headers by default, which is the safe failure mode. If an
   operator sets a hop count, the proxy must strip client-supplied headers first,
   or the limit key can be influenced by a caller.
4. **Anonymous access has no abuse ceiling beyond rate limiting.** By design for a
   research prototype. There is no authentication, so rate limiting is the only
   brake on `/predict`.
5. **A live browser pass has not been performed** (see H).
6. **Dependency vulnerability scanning is not in CI** (see F).
7. **CSP is not in report-only mode and not tuned against a deployed page.** It is
   compatible with the shipped Swagger/ReDoc assets; it has not been validated
   against a custom production frontend.
8. **`web/public/assets/heart-hero.png` is 2.43 MB** with no responsive variants.
   Performance, not security; unchanged from Stage 1.

## K. Deployment-dependent controls

Not implemented here, and why:

- HTTPS / TLS termination - needs a certificate listener
- HSTS preload - a registry submission
- WAF / DDoS - network edge
- Shared rate-limit store for multi-instance - see risk 1
- Production `ALLOWED_HOSTS` - the app cannot know its public hostname
- Reverse-proxy `X-Forwarded-For` stripping - the proxy owns that header
- Process supervision, log shipping, monitoring, alerting - platform concerns
- Secrets management - nothing secret is stored in this repository
- `pip-audit` / `npm audit` in CI - tooling

## L. Files changed

New:

- `app/security.py` - all runtime controls
- `tests/test_security.py` - 91 tests
- `tests/conftest.py` - raises the rate-limit budget for the functional suite only

Modified:

- `app/main.py` - `create_app` factory, docs control, model-info gating, minimal `/health`
- `app/model_service.py` - `build_model_info` filename-only artifact, gated environment block
- `app/schemas.py` - `ModelInfoResponse.environment` optional
- `README.md` - security section
- `SECURITY.md`, `PRODUCTION_SECURITY_CHECKLIST.md`, `OWASP_API_SECURITY_ASSESSMENT.md`,
  `API_INVENTORY.md`, `SECURITY_THREAT_MODEL.md`, `README_SECURITY_SECTION.md`

Unchanged: `data/`, `src/`, `models/`, `notebooks/`, `STAGE_3_AUDIT.md`,
`requirements.txt`, `web/` source, and all Stage 3 artifacts.

Retained unchanged as historical record: `SECURITY_AUDIT_PRE.md`.

## M. Final status

**Security Hardening Stage 1B: COMPLETE**

Runtime-enforced controls: **7** (security headers + HSTS, CORS, rate limiting,
request-size limits, request IDs, centralised safe errors, documentation control),
plus host/forwarded-header safety and `/health` + `/model-info` hardening.

No frozen ML artifact changed. All verification complete.

This report describes what the code in this repository does and what tests prove
about it. It makes no claim of regulatory compliance, certification, or clinical
validation.