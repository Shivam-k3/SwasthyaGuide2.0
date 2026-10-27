# OWASP API Security Top 10 (2023) - Assessment
**Repository:** Swasthyaguide2.0
**Date:** 2026-10-02 (Stage 1B: mitigations enforced)
**Phase:** 2 OWASP assessment, updated with implementation status

"Existing exposure" and "Evidence" below reflect Stage 1B. A control is only
credited as implemented where code enforces it and `tests/test_security.py` proves
it.

| Category | Applies | Explanation | Existing Exposure | Mitigation | Evidence | Residual Risk |
|----------|---------|-------------|-------------------|------------|----------|---------------|
| API1 - BOLA | No | No object IDs, no user-owned resources. Stateless single-patient scoring. | None | N/A - architecture has no objects to authorize | No endpoint takes resource IDs tied to users | NONE |
| API2 - Broken Auth | No | No auth by design (anonymous research tool). | None | N/A. Do not add auth. | `app` has no auth middleware; rate limiting is the only brake on `/predict` | LOW (no per-user ceiling exists because no users exist) |
| API3 - BOPLA | No | No complex objects with nested properties requiring different auth. `extra='forbid'` prevents property pollution. | None | Maintain strict schemas | `schemas.py` uses `extra='forbid'`, strict types | NONE |
| API4 - Unrestricted Resource Consumption | **Yes - now mitigated** | Was: no rate limits, no body size limits, `/predict` callable repeatedly. | Was: High. Now LOW single-instance | Rate limiting (endpoint-specific, 429 + `Retry-After`), 1 MiB body cap (413), declared and undeclared lengths | `RateLimitMiddleware` + `BodySizeLimitMiddleware`; live check: calls 1-20 = 200, calls 21-25 = 429 with `Retry-After: 58`; 1.5 MB body = 413; bounded counter table; `test_in_memory_limiter_can_be_replaced_by_another_implementation` | **MEDIUM if scaled out** - state is per process; needs a shared store first |
| API5 - Broken Function Level Auth | No | No admin/privileged functions. | None | N/A | Fixed small surface; no `/model` write or reload endpoints | NONE |
| API6 - Unrestricted Access to Sensitive Business Flows | Partial | Health prediction is sensitive; anonymous access intentional per design. | Public by design | Rate limiting, strict validation, size limits | Documented anonymous workflow; `/predict` 20/min per client | LOW |
| API7 - SSRF | No | No outbound requests to user URLs. Only local file I/O. | None | N/A | Code review confirms no URL fetching | NONE |
| API8 - Security Misconfiguration | **Yes - now mitigated** | Was: missing security headers, default CORS, uncontrolled prod docs exposure, no safe error handling. Was: Medium. Now LOW | Was: Medium. Now LOW | Security headers + production-only HSTS, explicit fail-closed CORS, env-controlled docs, centralised safe errors, startup logging of what is *not* configured | All responses carry the header set; no wildcard CORS code path; `/docs` 404 in production by default; unhandled exception returns a scrubbed envelope with a request ID; CSP validated against the shipped Swagger/ReDoc assets | LOW; depends on the operator setting `ALLOWED_HOSTS` |
| API9 - Improper Inventory Management | No | Small explicit API surface. No zombie routes. | None | Maintain API inventory | 3 functional endpoints + 3 documentation paths; see `API_INVENTORY.md` | NONE |
| API10 - Unsafe Consumption of APIs | No | Consumes no external APIs. | None | N/A | Backend is provider only; frontend consumes only this API | NONE |

## Supporting controls outside the Top 10

| Control | Status | Note |
|---|---|---|
| Rate-limit identity via `X-Forwarded-For` | Enforced | Forwarded headers ignored unless `SWASTHYAGUIDE_TRUSTED_PROXY_HOPS` is set; reads the Nth entry from the right. Replaces the Stage 1 stub that trusted the header unconditionally |
| Host header validation | Enforced when configured | `TrustedHostMiddleware` only when `ALLOWED_HOSTS` is set and has no `*`; a wildcard does not install it and does not pretend otherwise |
| Log injection via `X-Request-ID` | Enforced | IDs validated against `[A-Za-z0-9._~-]{1,64}`; malformed input replaced and logged |
| Information disclosure via `/health` and `/model-info` | Enforced | Minimal health shape; bare artifact filename; build-environment block withheld unless opted in |
| Dependency hygiene | No new dependency | `requirements.txt` unchanged, which `src/verify_stage3` depends on; `pip-audit` / `npm audit` not yet in CI |
| Authentication / authorization | Deliberately absent | Research prototype; adding it is a product decision |