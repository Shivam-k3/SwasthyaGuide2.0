# README Security Section
**Source text for the `Security & Privacy` section of `README.md`.**
**Status:** reflects Stage 1B. Every control listed here is enforced in code and
covered by `tests/test_security.py`.

## Security & Privacy

SWASTHYAGUIDE 2.0 is a research health risk assessment prototype. The following
security controls are active at runtime:

- **Stateless design:** no user accounts, sessions, databases, or health data
  persistence
- **Input validation:** strict server-side schema validation with unknown field
  rejection
- **Privacy-preserving:** no health inputs or predictions are logged; no browser
  storage of sensitive data
- **Security headers:** full header set on every response, including
  `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`,
  `Permissions-Policy` and `Content-Security-Policy`; HSTS in production only
- **CORS:** explicit origin allow-list from configuration; no wildcard is
  possible, and an unconfigured production deployment refuses all cross-origin
  requests
- **Rate limiting:** 20 requests/minute on `/predict`, 60/minute on other routes,
  600/minute on `/health`, with `429` and `Retry-After`
- **Request size limits:** 1 MiB default, returning `413`; enforced for bodies with
  or without a declared `Content-Length`
- **Request IDs:** every response returns `X-Request-ID`, and the ID appears in
  operational logs
- **Safe error handling:** no tracebacks, filesystem paths, dependency internals
  or environment details in any error response
- **Documentation control:** `/docs`, `/redoc` and `/openapi.json` are disabled in
  production by default
- **Frozen ML pipeline:** model artifacts and methodology are unchanged
  (Stage 3 validated, 69/69 checks)

Controls that belong to the deployment rather than the application - TLS
termination, shared rate-limit state for multi-instance deployments, `ALLOWED_HOSTS`,
trusted-proxy configuration, WAF, secrets management - are listed in
`PRODUCTION_SECURITY_CHECKLIST.md`.

**Medical safety:** this is research-only, not a medical device or diagnostic
system. See `MEDICAL_SAFETY_AND_LIMITATIONS.md`.

For security concerns, see `SECURITY.md`. This is a research prototype; production
deployments must complete the production security checklist.