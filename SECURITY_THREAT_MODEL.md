# SECURITY_THREAT_MODEL.md - Practical Threat Model
**Repository:** Swasthyaguide2.0
**Date:** 2026-10-02 (Stage 1B: mitigations enforced)
**Phase:** 1 threat model, with Stage 1B implementation status

"Existing control" now means enforced in code and covered by a test in
`tests/test_security.py`, not a proposal.

| Asset | Threat | Attack Surface | Existing Control | Mitigation | Residual Risk |
|-------|--------|----------------|------------------|------------|---------------|
| ML Prediction Service (`/predict`) | Automated API abuse / DoS | Public POST endpoint | Input validation (Pydantic, `extra='forbid'`), stateless, **endpoint rate limit 20/min with 429 + `Retry-After`**, **1 MiB body cap with 413** | Rate limiting, request size limits, timeouts | LOW single-instance; MEDIUM if scaled out with in-memory state |
| Model artifacts (`models/`) | Unauthorized access / info disclosure | Filesystem, not exposed via API | Not exposed; **`/model-info` returns a bare filename and withholds the build-environment block** | Never expose paths; do not add a static file mount for `models/` | VERY LOW |
| Health data inputs | Accidental logging / retention | Request payloads, logs | **Access log records method, path, status, duration, client IP, request ID only; query string dropped; no payload, input or result is ever logged** | Privacy-preserving logging enforced by `RequestIdMiddleware`; tested with a real payload | LOW |
| API surface (health, model-info, predict, docs) | Reconnaissance | Public endpoints | Minimal endpoints; **docs disabled in production by default**; **centralised safe errors**; **security headers on every response** | Prod docs control, safe error handling, no internal details | LOW |
| Rate-limit identity | Spoofed `X-Forwarded-For` evading limits | Request headers | **`client_ip()` ignores forwarded headers unless trusted proxy hops are declared; reads the Nth entry from the right** | Replaces the Stage 1 stub, which trusted `X-Forwarded-For` unconditionally | LOW, assuming the proxy strips client-supplied forwarded headers |
| Host header | Host-header injection / cache poisoning | Request headers | **`TrustedHostMiddleware` installed whenever `ALLOWED_HOSTS` is set and does not contain `*`**; a startup warning states when Host validation is off | Configure `ALLOWED_HOSTS` in production | MEDIUM if left unset by an operator |
| Log integrity | Request-ID injection forging log entries | Request headers | **Client-supplied IDs validated against `[A-Za-z0-9._~-]{1,64}`; malformed IDs replaced and the replacement logged** | Length cap and character allow-list | VERY LOW |
| Frontend bundle | Info disclosure via source | Static files | Built bundle; no secrets in `VITE_*`; **CSP enforced** | No secrets in frontend; CSP validated against the shipped docs assets | LOW |
| ML pipeline integrity | Tampering | Filesystem | Git tracked, frozen; `src/verify_stage3` asserts pinning | Verify checksums, git status checks | VERY LOW |
| Error messages | Info disclosure | API responses | **Centralised safe errors: no traceback, path, package internals or environment; validation errors drop `input`/`ctx`/`url`**; 503 message preserved byte-identical | Safe handlers + message scrubbing | LOW |
| Dependency supply chain | Known CVE exploitation | Installed packages | **No new runtime dependency added in Stage 1B**; `requirements.txt` unchanged | `pip-audit` / `npm audit` still not in CI | MEDIUM (scanning gap, not a runtime gap) |

## Residual risks worth stating plainly

1. **In-memory rate-limit state is per process.** N instances behind a balancer give
   a caller N times the configured budget. `security.RateLimiter` is a two-method
   interface so a shared-store implementation is a drop-in replacement. This is the
   main reason the `/predict` residual risk is not simply "LOW".
2. **No authentication, by design.** Anonymous research tool; rate limiting is the
   only brake on `/predict`. There is no per-user ceiling because there are no users.
3. **Host validation is off until `ALLOWED_HOSTS` is configured.** The application
   logs this rather than assuming it, but it remains an operator obligation.
4. **TLS, WAF and DDoS protection are edge concerns**, outside the application.
5. **No live browser pass** (no automation available), so DOM-level rendering and
   console behaviour are unverified.

Threats considered and not applicable: SSRF (no outbound requests from user
input), path traversal (no user-supplied file paths), injection into the ML
pipeline (input is coerced to a typed schema and passed to a persisted estimator),
and object-level authorization (the API has no object identifiers and no user-owned
resources).