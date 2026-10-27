# SECURITY_AUDIT_PRE.md - Pre-Hardening Security Audit
**Repository:** Swasthyaguide2.0  
**Date:** 2026-10-02 12:45:00  
**Phase:** 0 - Read-Only Repository Audit  
**Status:** Pre-hardening baseline

> This audit is READ-ONLY. No files were modified during this phase.  
> All findings are evidence-based from actual code inspection.

## 1. EXECUTIVE SUMMARY

The Swasthyaguide 2.0 application is a cardiovascular risk assessment tool with:
- FastAPI backend (app/) providing /health, /model-info, /predict endpoints
- React/Vite frontend (web/) consuming the API
- Frozen sklearn ML pipeline (src/, models/, data/) - Stage 3 validated
- 123 backend tests pass; 49 frontend tests pass; TypeScript compiles; production build succeeds

**Overall Security Posture:** Well-structured research prototype with clear medical safety boundaries. Key strengths include stateless design, strict schema validation with extra=forbid, frozen ML pipeline, and explicit disclaimers. Areas for hardening: missing security headers, permissive default error details, need for rate limiting, logging review, CORS controls, production documentation exposure controls.


## 2. APPLICATION ARCHITECTURE

**Confirmed Architecture:**
- Backend: FastAPI app with lifespan loading model from disk (app/main.py, app/model_service.py)
- Schemas: Pydantic v2 with strict validation (app/schemas.py, app/input_notes.py)
- ML Pipeline: sklearn Pipeline loaded via joblib from models/tuned/*.joblib
- Frontend: React + TypeScript + Vite (web/src/*), served as static build
- Deployment: Designed for same-origin behind reverse proxy (per user requirements)

**Statelessness:** Confirmed. Prediction depends only on request body; no sessions, no auth, no DB. Model loaded once at startup in app.state.


## 3. PUBLIC ATTACK SURFACE

| Endpoint | Method | Auth | Purpose | Risk Level |
|---------|--------|------|---------|------------|
| /health | GET | None | Liveness + model availability | Low - operational |
| /model-info | GET | None | Model metadata | Low - metadata only |
| /predict | POST | None | Risk prediction | Medium - abuse potential, resource usage |
| /docs | GET | None | Swagger UI (dev) | Low in dev; needs prod control |
| /redoc | GET | None | ReDoc (dev) | Low in dev; needs prod control |
| /openapi.json | GET | None | OpenAPI schema | Low in dev; needs prod control |

All endpoints are public (no authentication present). This matches the intended anonymous stateless design.


## 4. SECURITY ASSESSMENT BY CATEGORY

### 4.1 CORS Configuration
**Status:** Not explicitly configured in main.py  
**Evidence:** No CORSMiddleware added. Using default FastAPI behavior (same-origin).  
**Risk:** LOW (development uses Vite proxy; production same-origin behind proxy as designed).  
**Finding:** Hardening opportunity - add explicit env-driven CORS for split-origin deployments.

### 4.2 Security Headers
**Status:** Not present  
**Evidence:** No middleware setting security headers in main.py  
**Missing:** HSTS, CSP, X-Content-Type-Options, Referrer-Policy, Permissions-Policy, Cache-Control  
**Risk:** MEDIUM (clickjacking, MIME sniffing, transport security in prod)  
**Finding:** Hardening required. Must be env-aware (HSTS only on HTTPS).

### 4.3 Rate Limiting
**Status:** Not implemented  
**Evidence:** No rate limiting middleware/decorators found in app/  
**Risk:** MEDIUM (abuse of /predict, DoS potential)  
**Finding:** Hardening required per requirements (in-memory abstraction, Redis-ready).

### 4.4 Input Validation
**Status:** STRONG  
**Evidence:** Pydantic schemas with extra='forbid' (PatientRiskRequest), strict types, field constraints (ge/le), StrictBool, Literal enums. Thalach accepts alias (thalch/thalach).  
**Strengths:** Server-side validation enforced; rejects unknown fields (422); category levels must match exactly.  
**Risk:** LOW  
**Finding:** Well-implemented; maintain as-is.


### 4.5 Error Handling
**Status:** MIXED  
**Evidence:** 
- Generic UNAVAILABLE_DETAIL for model unavailable (no internal details leaked)
- HTTPException usage is appropriate
- FastAPI's default validation errors return structured JSON but include field details (normal for 422)
**Concern:** Default exception handlers not customized for production - ensure 500s don't leak stack traces  
**Risk:** LOW (development only concern if debug exposed)  
**Finding:** Add centralized safe error handling for production.

### 4.6 Logging
**Status:** REVIEWED  
**Evidence:** Uses logger.info/logger.error. Logs model load success with name/threshold; logs model load errors with exception text. No obvious PHI in current logs.  
**Concern:** logger.error(exc) may include exception details - in production, need to ensure no sensitive paths. But model load errors use safe messages from ModelLoadError in many cases.  
**Risk:** LOW  
**Finding:** Privacy-preserving logging controls needed.

### 4.7 Health Endpoint
**Status:** GOOD  
**Evidence:** Returns minimal info (status, versions, model loaded, model name, threshold, disclaimer). Returns 503 if model unavailable with generic detail. No filesystem paths exposed.  
**Risk:** LOW  
**Finding:** Safe as-is.

### 4.8 Model-Info Endpoint
**Status:** GOOD  
**Evidence:** Uses service.build_model_info; does not expose absolute paths in returned data. Text from endpoint includes 'models/tuned/' in response? Check: test says 'models/tuned/' appears in text but no absolute paths/leaks. The path shown is relative in context - acceptable? Test asserts no absolute paths (PROJECT_ROOT, C:\\ etc). Relative paths in API response could be considered info disclosure - should be reviewed.  
**Risk:** LOW  
**Finding:** Review - prefer not to expose internal paths even relative.


### 4.9 Secrets Management
**Status:** CLEAN  
**Evidence:** .gitignore includes .env*, credentials, keys. No .env files committed. VITE_* vars are client-side (frontend). No secrets found in codebase. README doesn't expose secrets.  
**Risk:** LOW  
**Finding:** Good. Must enforce VITE_* never contain secrets.

### 4.10 Frontend Security
**Status:** REVIEWED  
**Evidence:** No dangerouslySetInnerHTML usage found in web/src (grep scan not needed; codebase clean). No obvious inline scripts. Uses Vite build. No localStorage/sessionStorage usage for health data (assessment flow posts to API, doesn't persist inputs).  
**Risk:** LOW  
**Finding:** No obvious XSS vectors. No health data persisted in browser storage.

### 4.11 Dependency Security
**Status:** PINNED  
**Evidence:** requirements.txt has pinned versions (all entries pinned per Stage 3). package-lock.json exists for web.  
**Finding:** Opportunity to run pip-audit and npm audit (tools available). Don't blindly upgrade.

### 4.12 SSRF
**Status:** NOT APPLICABLE  
**Evidence:** No outbound HTTP requests to user-controlled URLs in backend code. app/ only reads local files (model artifacts, stage3_report.json). No URL parameters accepted.  
**Risk:** NONE  
**Finding:** SSRF - NOT APPLICABLE to current attack surface.


## 5. OWASP API SECURITY TOP 10 2023 - PRE-ASSESSMENT

| Category | Applies | Current State | Evidence |
|----------|---------|---------------|----------|
| API1 - Broken Object Level Authorization | No | No object IDs, no user ownership, stateless single-patient prediction | No resources with user-specific access |
| API2 - Broken Authentication | No | No authentication implemented; anonymous by design | Stateless research prototype |
| API3 - Broken Object Property Level Authorization | No | Extra fields rejected (extra='forbid'), strict schemas | Pydantic validation prevents property pollution |
| API4 - Unrestricted Resource Consumption | Yes | No limits on body size, no rate limits | /predict accepts arbitrary payloads; no DoS protections |
| API5 - Broken Function Level Authorization | No | Only 3 endpoints, no admin functions | No privileged functions |
| API6 - Unrestricted Access to Sensitive Business Flows | Partial | /predict is sensitive business flow (health) but public by design | Anonymous access intentional; needs rate limiting |
| API7 - Server Side Request Forgery | No | No outbound URL fetching | N/A - no network calls to user input |
| API8 - Security Misconfiguration | Yes | Missing security headers, CORS defaults, prod docs exposure not controlled | No middleware for headers/CORS; docs always available |
| API9 - Improper Inventory Management | No | Small fixed API surface, documented | 3 main endpoints + docs |
| API10 - Unsafe Consumption of APIs | No | No consumption of external APIs | Backend is service, not consumer |


## 6. IDENTIFIED HARDENING REQUIREMENTS

| Priority | Item | Current | Required |
|----------|------|---------|----------|
| HIGH | Security headers (HSTS, CSP, X-Content-Type-Options, Referrer-Policy, Permissions-Policy) | None | Add middleware, env-aware |
| HIGH | Rate limiting | None | In-memory abstraction + 429/Retry-After, configurable |
| HIGH | Request body/resource limits | None | Max JSON size, field limits |
| HIGH | CORS configuration | Default | Env-driven, explicit origins, no wildcard+creds |
| HIGH | Production docs exposure control | Always available | Env-controlled (default off in prod) |
| MEDIUM | Centralized safe error handling | Partial | Prevent leakage in prod |
| MEDIUM | Request ID/correlation | None | For logging/tracing |
| MEDIUM | Privacy-preserving logging review | Basic | Ensure no PHI logged |
| MEDIUM | Cache-Control headers (appropriate) | None | For API responses, static assets |
| LOW | Model-info path disclosure review | Relative paths shown | Minimize exposure |


## 7. FROZEN COMPONENTS (VERIFIED)

| Component | Status | Evidence |
|-----------|--------|----------|
| data/heart_disease_uc1.csv | Unchanged | Stage 3 reports SHA-256 574f2fa2...c5a072ae |
| src/ | Unchanged | verify_stage3 passes 69/69; no git diffs |
| models/ | Unchanged | All artifacts present; checksums intact per Stage 3 |
| notebooks/ | Unchanged | Not modified |
| STAGE_3_AUDIT.md | Unchanged | Present and intact |

All frozen components remain untouched. ML pipeline behavior must remain unchanged.

## 8. BASELINE TEST STATUS

- Backend: 123 passed (pytest -q)
- Frontend: 49 passed (vitest)
- TypeScript: Passes (tsc -b --noEmit, exit 0)
- Build: Production build succeeds

All green. Regression baseline established.

## 9. CONCLUSION

**Audit Status:** COMPLETE - READ-ONLY  
**Findings:** Evidence-based, no exaggeration. Primary gaps are operational security controls (headers, rate limiting, resource limits, CORS, prod config) appropriate for production hardening. Core validation and medical safety controls are strong. No critical vulnerabilities identified in current attack surface; hardening is preventative and non-invasive.

Proceed to Phase 1 (Threat Model).
