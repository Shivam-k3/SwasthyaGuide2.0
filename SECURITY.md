# SECURITY.md - Security Architecture & Controls
**Repository:** Swasthyaguide2.0
**Date:** 2026-10-02 (Stage 1B: runtime implementation)
**Last verified against code:** Stage 1B

## 1. Security Architecture

Stateless FastAPI backend. Same-origin deployment behind an HTTPS reverse proxy is
the intended production shape; split-origin is supported through explicit
configuration. No authentication, sessions, database, or persistence, by design.

Every control below runs at the application boundary, in `app/security.py` and
`app/main.py`. None of them touches the frozen ML pipeline: `data/`, `src/`,
`models/`, `notebooks/` and the Stage 3 artifacts are byte-identical, and
`python -m src.verify_stage3` still reports 69/69.

## 2. Control status

The distinction that matters: a control is only listed as IMPLEMENTED if code
enforces it and a test proves it.

### IMPLEMENTED IN APPLICATION (runtime-enforced, tested)

| Control | Implementation | Proof |
|---|---|---|
| Security headers (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`, `Content-Security-Policy`, `Cross-Origin-*`, `X-DNS-Prefetch-Control`) | `app/security.py` `SecurityHeadersMiddleware` | `tests/test_security.py::test_security_headers_are_present_on_a_normal_response` |
| HSTS, only when HTTPS/production is in force | `SecurityHeadersMiddleware` + `SecuritySettings.hsts_enabled` | `test_hsts_is_absent_during_local_http_development`, `test_hsts_is_present_in_production_by_default`, `test_hsts_is_present_when_forced_https` |
| `Cache-Control: no-store` on API responses | `SecurityHeadersMiddleware` | `test_cache_control_prevents_intermediary_caching` |
| Explicit, environment-driven CORS. No wildcard code path; fail-closed when unconfigured in production | `SecuritySettings.from_env`, `cors_options`, `install_security` | `test_allowed_origin_receives_cors_headers`, `test_disallowed_origin_receives_no_cors_grant`, `test_wildcard_cannot_be_configured`, `test_production_without_configured_origins_refuses_cross_origin` |
| CORS preflight handling | `starlette.middleware.cors.CORSMiddleware` | `test_preflight_from_an_allowed_origin_is_answered`, `test_preflight_from_a_disallowed_origin_is_refused` |
| Rate limiting, per endpoint, with 429 and `Retry-After` | `RateLimitMiddleware` + `InMemoryRateLimiter` | `test_predict_is_rate_limited_and_returns_429_with_retry_after`, `test_rate_limit_budget_recovers_within_the_window` |
| Rate-limit abstraction that a Redis implementation can satisfy without touching call sites | `security.RateLimiter` interface | `test_in_memory_limiter_can_be_replaced_by_another_implementation` |
| Request body size limit, declared and undeclared length, returning 413 | `BodySizeLimitMiddleware` | `test_oversized_body_is_rejected_with_413`, `test_chunked_body_without_content_length_is_also_capped` |
| Request IDs: generated, validated on ingress, returned in `X-Request-ID`, present in operational logs | `RequestIdMiddleware`, `validate_request_id`, `RequestIdFilter` | `test_every_response_carries_a_request_id`, `test_a_valid_client_supplied_request_id_is_echoed`, `test_a_malformed_client_request_id_is_replaced_not_echoed`, `test_the_request_id_reaches_the_logs` |
| No health input, payload, or result in logs | access log records method/path/status/duration/id only | `test_logs_never_contain_the_prediction_payload` |
| Centralised safe error handling: no traceback, path, package internals, or env | `SafeErrorMiddleware`, `safe_validation_exception_handler`, `safe_http_exception_handler`, `_scrub_message` | `test_an_internal_exception_returns_a_safe_envelope_not_a_traceback`, `test_internal_errors_leak_no_internals` |
| Validation errors keep the field and the reason, drop `input`/`ctx`/`url` | `safe_validation_exception_handler` | `test_validation_errors_keep_the_field_and_the_reason`, `test_validation_errors_do_not_echo_the_submitted_values` |
| Intentional 400/422/503 behaviour preserved | handlers pass existing details through | `test_503_behaviour_is_preserved_when_the_model_is_unavailable`, `test_malformed_json_still_returns_422_without_internals` |
| API documentation environment-controlled; disabled in production by default, opt-in supported | `create_app` `docs_url`/`redoc_url`/`openapi_url` | `test_documentation_is_available_in_development`, `test_documentation_is_disabled_in_production_by_default`, `test_production_documentation_can_be_re_enabled_explicitly`, `test_the_openapi_schema_implementation_is_retained_not_deleted` |
| Forwarded headers not trusted by default | `client_ip(scope, trusted_proxy_hops)` | `test_forwarded_headers_are_not_trusted_by_default`, `test_spoofed_forwarded_headers_cannot_dodge_the_rate_limit` |
| Host validation when `ALLOWED_HOSTS` is configured | `TrustedHostMiddleware`, installed conditionally | `test_host_validation_rejects_an_unknown_host_when_configured`, `test_a_wildcard_host_list_does_not_silently_disable_validation` |
| `/health` minimal; no paths, versions, environment | `app/main.py` `health()` | `test_health_exposes_only_minimal_operational_state`, `test_health_leaks_no_paths_versions_or_environment` |
| `/model-info`: bare artifact filename, no directory component | `app/model_service.py` `build_model_info` | `test_model_info_artifact_is_a_bare_filename`, `test_model_info_leaks_no_absolute_paths` |
| `/model-info`: Stage 3 build environment withheld by default | `include_build_environment` + `SWASTHYAGUIDE_EXPOSE_BUILD_ENV` | `test_model_info_withholds_the_build_environment_by_default` |

### DEPLOYMENT-DEPENDENT (documented, cannot be enforced inside the application)

| Item | Why it cannot live in the app | Where it is done |
|---|---|---|
| TLS termination | Requires a listening endpoint for certificates | Reverse proxy / load balancer |
| HSTS preload submission | Registry submission, outside the app | Operator |
| WAF / DDoS protection | Network edge | CDN / provider |
| Multi-instance rate-limit state | **In-memory limiting is per process.** N instances behind a balancer give a caller N x the configured limit. A Redis (or equivalent shared) implementation of `security.RateLimiter` is required before scaling out | Operator, with `app/security.py` interface |
| Production hostname and trusted-proxy config | The app cannot know its own public hostname | `ALLOWED_HOSTS`, `SWASTHYAGUIDE_TRUSTED_PROXY_HOPS` |
| Forwarded-header stripping at the edge | The proxy must remove client-supplied `X-Forwarded-For` before setting its own, otherwise the hop count is meaningless | Reverse proxy config |
| Process supervision and log shipping | Operationally outside the app | Platform |
| Secrets management | No secret is stored in this repository | Platform secret store |
| Dependency vulnerability scanning | Tooling, not runtime code | `pip-audit` / `npm audit` in CI |

The application logs, at startup, exactly which of these are not configured (see
`security.log_configuration`). A control that is not enforced is stated in the
startup log rather than left to be assumed.

### FUTURE (not started)

- Image-upload security (pre-requisites already scoped in
  `FUTURE_MEDICAL_IMAGE_SECURITY_REQUIREMENTS.md`). Image upload is not part of
  this application, so no upload limits are implemented.
- Authentication. Deliberately absent: this is an anonymous research prototype.
  Adding it would be a product decision, not a hardening step.
- Redis-backed rate limiting, when the deployment becomes multi-instance.
- Dependency scanning wired into CI.

## 3. Configuration reference

All variables are optional. Defaults are chosen so that `uvicorn app.main:app`
is safe in development and fail-closed in production.

| Variable | Default | Effect |
|---|---|---|
| `SWASTHYAGUIDE_ENV` | `development` | `production` turns on HSTS, disables `/docs` by default, and makes unset CORS fail closed |
| `SWASTHYAGUIDE_ENABLE_DOCS` | on in dev, off in production | Explicit override for Swagger/ReDoc/OpenAPI. `ENABLE_DOCS` accepted as a legacy alias |
| `CORS_ALLOWED_ORIGINS` | localhost/127.0.0.1 :5173 and :8000 in dev; **empty in production** | Comma-separated explicit origins. `*` is rejected |
| `CORS_ALLOW_CREDENTIALS` | `false` | Rejected at configuration time if enabled with no explicit origins |
| `SWASTHYAGUIDE_FORCE_HTTPS` | `false` (on in production) | Emits HSTS. Off locally so http://localhost is not pinned |
| `ALLOWED_HOSTS` | unset (validation off, warning logged) | Comma-separated host allow-list for `TrustedHostMiddleware` |
| `SWASTHYAGUIDE_TRUSTED_PROXY_HOPS` | `0` | Number of trusted proxies. `0` means forwarded headers are ignored entirely |
| `MAX_REQUEST_SIZE_BYTES` | `1048576` | Body budget; larger bodies get 413 |
| `SWASTHYAGUIDE_RATE_LIMIT_ENABLED` | `true` | Master switch |
| `RATE_LIMIT_PER_MINUTE` | `60` | Site-wide per-client budget |
| `PREDICT_RATE_LIMIT_PER_MINUTE` | `20` | `/predict` budget. Deliberately tighter |
| `HEALTH_RATE_LIMIT_PER_MINUTE` | `600` | `/health` budget. Deliberately generous so monitoring is not mistaken for abuse |
| `SWASTHYAGUIDE_CONTENT_SECURITY_POLICY` | restrictive default in `security.DEFAULT_CSP` | Must remain compatible with the shipped Swagger UI / ReDoc assets |
| `SWASTHYAGUIDE_EXPOSE_BUILD_ENV` | `false` | Whether `/model-info` republishes the Stage 3 build-environment block |

## 4. Medical Safety

This is a research prototype for health risk assessment, not a medical device and
not a diagnostic system. Outputs are model-estimated risk scores. Emergency
situations must not be diagnosed by the model; users should seek emergency care.
No treatment or medication advice is given.

No statement in this repository should be read as a claim of regulatory
compliance or clinical validation.

## 5. Reporting a concern

See `SECURITY_INCIDENT_RESPONSE.md`. This is a research prototype with no
published support commitment and no vulnerability disclosure programme.