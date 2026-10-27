# API Inventory
**Date:** 2026-10-02 (Stage 1B: runtime enforcement)

"Rate limit" now means enforced in code, not proposed.

| Endpoint | Method | Auth Required | Authz Required | Request Schema | Response Schema | Rate Limit (enforced) | Sensitive Data | Production Exposure |
|----------|--------|---------------|----------------|----------------|-----------------|-----------------------|----------------|---------------------|
| /health | GET | No | No | None | HealthResponse | 600/min per client (own budget, deliberately generous) | No PHI | Required (monitoring) |
| /model-info | GET | No | No | None | ModelInfoResponse | 60/min per client (site default) | No PHI | Optional; safe public metadata only |
| /predict | POST | No | No | PatientRiskRequest | RiskPredictionResponse | **20/min per client**; 429 + `Retry-After` when exceeded | Health inputs, transient only | As needed |
| /docs | GET | No | No | N/A | HTML | 60/min per client | N/A | **Disabled by default in production** (`SWASTHYAGUIDE_ENABLE_DOCS`) |
| /redoc | GET | No | No | N/A | HTML | 60/min per client | N/A | **Disabled by default in production** |
| /openapi.json | GET | No | No | N/A | JSON | 60/min per client | N/A | **Disabled by default in production** |

`OPTIONS` preflight is exempt from rate limiting so a browser's preflight does not
consume a caller's prediction budget.

## Cross-cutting controls

| Control | State | Notes |
|---|---|---|
| Body size limit | 1 MiB (`MAX_REQUEST_SIZE_BYTES`) | 413 before prediction work; undeclared-length bodies also capped |
| `X-Request-ID` | Every response | Client-supplied IDs validated against `[A-Za-z0-9._~-]{1,64}` |
| Security headers | Every response | HSTS only in production/HTTPS |
| CORS | Explicit origins only | Empty allow-list in production refuses all cross-origin |
| Error shape | `{"detail": ..., "error": {code, message, request_id}}` | No traceback, path, package internals or environment |

## Notes

- `/model-info` returns a bare artifact filename and withholds the Stage 3 build
  environment block unless `SWASTHYAGUIDE_EXPOSE_BUILD_ENV=true`.
- `/health` returns only status, API stage and version, model availability, the
  served threshold and the disclaimer.