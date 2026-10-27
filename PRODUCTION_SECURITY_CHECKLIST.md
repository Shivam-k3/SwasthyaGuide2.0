# PRODUCTION_SECURITY_CHECKLIST.md
**Updated:** 2026-10-02 (Stage 1B)

Two lists: what the application now enforces by itself, and what an operator must
still do. A box in the first list means code enforces it and a test proves it; see
`SECURITY.md` for the test names and `SECURITY_HARDENING_REPORT.md` for evidence.

## Enforced by the application (verified by tests)

- [x] Security headers on every response: `X-Content-Type-Options`, `X-Frame-Options`,
      `Referrer-Policy`, `Permissions-Policy`, `Content-Security-Policy`,
      `Cross-Origin-Opener-Policy`, `Cross-Origin-Resource-Policy`,
      `X-DNS-Prefetch-Control`, `Cache-Control: no-store`
- [x] HSTS on production/HTTPS only; deliberately absent on local HTTP
- [x] Explicit environment-driven CORS; wildcard impossible; unconfigured production
      refuses all cross-origin rather than allowing everything
- [x] Rate limiting on `/predict` (20/min), `/health` (600/min), all routes (60/min),
      configurable; 429 with `Retry-After`
- [x] Request body size limit (1 MiB default); 413 before any prediction work
- [x] Request IDs returned in `X-Request-ID` and present in operational logs;
      client-supplied IDs validated before use
- [x] Centralised safe error handling: no traceback, path, package internals or
      environment in any response
- [x] Validation errors keep the field and reason, never echo submitted values
- [x] `/docs`, `/redoc`, `/openapi.json` disabled in production by default, opt-in
      supported, schema retained in-process
- [x] Forwarded headers ignored unless trusted proxy hops are declared
- [x] `/health` minimal; `/model-info` free of absolute paths and of the build
      environment block
- [x] No PHI, request body or prediction result in logs

## Operator must do (cannot be enforced inside the application)

- [ ] **TLS with a valid certificate; HTTP redirected to HTTPS.** The application
      does not terminate TLS.
- [ ] Set `ALLOWED_HOSTS` to the public hostname(s). Until this is set, Host header
      validation is disabled and the application logs a warning saying so.
- [ ] Set `CORS_ALLOWED_ORIGINS` to the explicit production frontend origin(s).
      Same-origin deployment needs no CORS; split-origin does.
- [ ] Set `SWASTHYAGUIDE_TRUSTED_PROXY_HOPS` to the real number of proxies, **and**
      configure the proxy to strip client-supplied `X-Forwarded-For` before setting
      its own. Without the hop count the application ignores the header entirely,
      which is safe but means every client shares one rate-limit bucket.
- [ ] **Do not scale out while rate limiting is in-memory.** State is per process;
      N instances give a caller N times the budget. Move `security.RateLimiter` to
      a shared store first.
- [ ] Review rate-limit values against real traffic (`RATE_LIMIT_PER_MINUTE`,
      `PREDICT_RATE_LIMIT_PER_MINUTE`, `HEALTH_RATE_LIMIT_PER_MINUTE`).
- [ ] Review `MAX_REQUEST_SIZE_BYTES` against real payload sizes.
- [ ] Confirm `SWASTHYAGUIDE_ENABLE_DOCS` is not set in production unless intended.
- [ ] Confirm `SWASTHYAGUIDE_EXPOSE_BUILD_ENV` is not set (it republishes the OS
      build string and dependency versions).
- [ ] Secrets stored outside the repository; `VITE_*` variables contain no secrets.
- [ ] Secret scan completed.
- [ ] Dependency audit completed (`pip-audit`, `npm audit`) - not yet in CI.
- [ ] `CSP` validated against the actual deployed frontend, not just the shipped
      Swagger/ReDoc pages.
- [ ] Consider CSP `report-only` rollout before enforcing.
- [ ] Process supervision, log shipping, monitoring and alerting configured.
- [ ] Reverse proxy / WAF / load balancer controls reviewed.
- [ ] Browser storage reviewed: the assessment flow posts to the API and persists
      no health data in the browser.
- [ ] Privacy notice and medical disclaimer in place.
- [ ] Model limitations documented and current.
- [ ] Security tests and all regression tests pass.
- [ ] Frozen ML verification passes: dataset and model checksums unchanged, 69/69.

## Verification commands

```bash
python -m pytest -q                          # expect 214 passed
python -m pytest tests/test_security.py -q   # expect 91 passed
python -m src.verify_stage3                  # expect 69/69 checks passed
cd web && npm test                           # expect 49 passed
cd web && npx tsc -b --noEmit                # expect exit 0
cd web && npm run build                      # expect success
```