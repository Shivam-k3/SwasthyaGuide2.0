# PRIVACY_LIFECYCLE.md
**Date:** 2026-10-02 (Stage 1B: runtime logging behaviour verified)

## Data Lifecycle

INPUT → VALIDATION → PROCESSING (in-memory, stateless) → PREDICTION → RESPONSE → DISPOSAL (immediate)

- Prediction inputs are NOT persisted by default
- Prediction results are NOT persisted by default
- No health data stored in browser storage (localStorage/sessionStorage/cookies)
- No health data sent to analytics
- No health data logged
- No third-party sharing by default
- No patient database; no user accounts; no health record persistence

Temporary files (if any) must use isolated temp storage, restrict permissions, avoid
public dirs, delete after processing with cleanup on exceptions.

## Runtime enforcement (Stage 1B)

These are no longer claims about intent; each has a test in
`tests/test_security.py`.

| Statement | Enforcement |
|---|---|
| No health data logged | `RequestIdMiddleware` logs method, path, status, duration, client address and request ID only. The body is never passed to a logger. The query string is dropped, so URL-borne data is not logged either. `test_logs_never_contain_the_prediction_payload` posts a real payload and asserts the cholesterol value, chest-pain category and response field names are absent from the captured log text |
| Values not echoed in errors | Validation errors keep `type`, `loc` and `msg` and drop `input` and `ctx`, so a caller's own health values are not reflected in the response body |
| Errors carry no server internals | Traceback, filesystem path, package directory and environment are all excluded; only a code, a scrubbed message and the request ID are returned |
| No persistence | No database, no cache, no session. Rate-limit counters are keyed by client address and hold counts, never request content, and the table is bounded |
| Browser-side retention | The assessment form posts to the API and keeps no health data in browser storage. Verified by inspection of `web/src`; the API is not reachable cross-origin by default |

## Notes for operators

- The client IP address is logged, which is personal data under GDPR. Retain logs
  for as short a period as operational need allows, and treat log storage as a
  personal-data store.
- `X-Request-ID` is client-influenceable only within a validated character set
  (`[A-Za-z0-9._~-]{1,64}`), so it cannot be used to inject content into logs.
- Supply-chain: no new runtime dependency was added; `requirements.txt` is
  unchanged. `pip-audit` / `npm audit` are not yet part of CI.