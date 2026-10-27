# SECURITY_INCIDENT_RESPONSE.md
**Date:** 2026-10-02 (Stage 1B: request IDs and 429 telemetry now available)

## 1. Detection & Triage
- Monitor for unusual traffic patterns (rate limit 429 spikes), errors, availability issues
- Correlate on `X-Request-ID`: every response carries one and it appears in the
  access log, so a report of a specific failed request can be traced without
  retaining payloads
- Note that 429 and 413 responses are *not* errors to chase individually: they are
  the rate limiter and the body-size limit doing their job. Sustained 429s mean the
  budget is too low for real traffic or something is looping; sustained 413s mean
  a client is sending oversized bodies
- Classify: operational, security, data/privacy

## 2. Containment, Investigation, Recovery
- Contain affected instances, preserve security-relevant logs (without PHI), assess scope
- Preserve the request IDs from the incident window before log retention expires;
  they are the only correlation handle available, since no request body is logged
- Rotate credentials if any exposed (none expected), remove compromised components
- Recover to known-good state; verify frozen ML integrity
  (`python -m src.verify_stage3`, expecting 69/69, plus the dataset and model checksums)

## 3. Post-Incident Review
- Document timeline, root cause, actions, preventive measures
- Update controls as needed, and record which controls are enforced in code versus
  which remain deployment-dependent (see `SECURITY.md`)

## 4. Limitation to state in any incident record

This is a research prototype with no authentication, no persistence and no
monitoring infrastructure. There is no audit trail of who called the API beyond an
IP address and a request ID, no session to revoke, and no disclosure programme. An
incident cannot be attributed to a user from application data alone, and cannot be
unwound because nothing is stored.

