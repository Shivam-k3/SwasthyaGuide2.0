"""
SWASTHYAGUIDE 2.0 - runtime security controls.

Stage 1B of security hardening. Stage 1 produced the audit, the threat model and
the OWASP assessment; those documents described controls that were *designed* but
not *enforced*. This module is where they become runtime behaviour, at the
application boundary only.

Every control here sits outside the ML boundary. Nothing in this file reads a
model artifact, touches a threshold, or alters a prediction: `src/`, `models/`,
`data/` and `notebooks/` are unchanged, and `app/model_service.py` keeps serving
the identical persisted pipeline at the identical operating point.

Controls implemented here
-------------------------
1.  Security headers, with HSTS only when HTTPS/production is in force
2.  Environment-driven explicit CORS (never wildcard, fail-closed in production)
3.  Rate limiting behind a Redis-swappable abstraction, per-endpoint
4.  Request body size limits, rejected before prediction work happens
5.  Request/correlation IDs, validated on ingress, returned and logged
6.  Centralised safe error envelopes with no internal detail
7.  Environment-controlled API documentation exposure
8.  Forwarded-header and Host validation, configured rather than assumed

Deployment-dependent, NOT enforced here
---------------------------------------
TLS termination, HSTS preload, WAF, multi-instance rate-limit state, and
production hostname/proxy configuration cannot be enforced from inside the
application. See PRODUCTION_SECURITY_CHECKLIST.md.

No new dependency is required for any of this: it is standard library plus the
FastAPI/Starlette middleware that `requirements.txt` already pins.
"""

from __future__ import annotations

import contextvars
import logging
import os
import re
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping, MutableMapping

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

logger = logging.getLogger("swasthyguide.api")

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

# --------------------------------------------------------------------------- #
# Environment variable names - one place, so the documentation can cite them
# --------------------------------------------------------------------------- #

ENV_APP_ENV = "SWASTHYAGUIDE_ENV"
ENV_ENABLE_DOCS = "SWASTHYAGUIDE_ENABLE_DOCS"
ENV_CORS_ORIGINS = "CORS_ALLOWED_ORIGINS"
ENV_CORS_CREDENTIALS = "CORS_ALLOW_CREDENTIALS"
ENV_FORCE_HTTPS = "SWASTHYAGUIDE_FORCE_HTTPS"
ENV_ALLOWED_HOSTS = "ALLOWED_HOSTS"
ENV_TRUSTED_PROXY_HOPS = "SWASTHYAGUIDE_TRUSTED_PROXY_HOPS"
ENV_MAX_BODY_BYTES = "MAX_REQUEST_SIZE_BYTES"
ENV_RATE_LIMIT_ENABLED = "SWASTHYAGUIDE_RATE_LIMIT_ENABLED"
ENV_RATE_LIMIT_DEFAULT = "RATE_LIMIT_PER_MINUTE"
ENV_RATE_LIMIT_PREDICT = "PREDICT_RATE_LIMIT_PER_MINUTE"
ENV_RATE_LIMIT_HEALTH = "HEALTH_RATE_LIMIT_PER_MINUTE"
ENV_CSP = "SWASTHYAGUIDE_CONTENT_SECURITY_POLICY"
ENV_EXPOSE_BUILD_ENV = "SWASTHYAGUIDE_EXPOSE_BUILD_ENV"

#: Legacy alias kept because the Stage 1 documentation named it.
ENV_ENABLE_DOCS_LEGACY = "ENABLE_DOCS"

PRODUCTION = "production"
DEVELOPMENT = "development"

REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_HEADER_LOWER = b"x-request-id"

#: Inbound IDs are accepted only if they look like a short opaque token. Anything
#: else is replaced with a freshly generated UUID, so a client cannot inject
#: newlines, control characters or log-forging sequences into our logs through
#: the correlation header.
VALID_REQUEST_ID = re.compile(r"\A[A-Za-z0-9._~-]{1,64}\Z")

MAX_INBOUND_REQUEST_ID_LENGTH = 64

# --------------------------------------------------------------------------- #
# Header values
# --------------------------------------------------------------------------- #

DEFAULT_CSP = "; ".join(
    (
        # The API itself only ever returns JSON. Locking the default to 'none'
        # means a page that manages to execute markup here has no source to load
        # anything from.
        "default-src 'none'",
        "frame-ancestors 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "object-src 'none'",
        # /docs and /redoc ship from FastAPI itself: inline bootstrap script and
        # style plus the jsdelivr CDN copy of swagger-ui / redoc. When the docs
        # are disabled (production default) neither source is reachable anyway.
        "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'",
        "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'",
        "img-src 'self' data:",
        "font-src 'self' data: https://cdn.jsdelivr.net",
        "connect-src 'self'",
    )
)

DEFAULT_PERMISSIONS_POLICY = (
    "accelerometer=(), camera=(), display-capture=(), encrypted-media=(), "
    "geolocation=(), gyroscope=(), magnetometer=(), microphone=(), "
    "midi=(), payment=(), usb=()"
)

HSTS_VALUE = "max-age=31536000; includeSubDomains"

#: Nothing this API returns is cacheable: a prediction response is a health
#: result and must not be written to a shared cache by an intermediary.
API_CACHE_CONTROL = "no-store"

# --------------------------------------------------------------------------- #
# Error envelope
# --------------------------------------------------------------------------- #

#: Machine-readable codes. Stable, so a client can branch without string
#: matching on prose.
CODE_VALIDATION = "VALIDATION_ERROR"
CODE_RATE_LIMITED = "RATE_LIMITED"
CODE_PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
CODE_INTERNAL = "INTERNAL_ERROR"
CODE_MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
CODE_BAD_HOST = "INVALID_HOST"
CODE_HTTP_ERROR = "HTTP_ERROR"

GENERIC_INTERNAL_MESSAGE = "An unexpected error occurred."
GENERIC_HTTP_MESSAGE = "The request could not be completed."
GENERIC_RATE_LIMIT_MESSAGE = "Too many requests. Please slow down and retry shortly."
GENERIC_PAYLOAD_MESSAGE = "The request body is larger than this service accepts."

#: Patterns that indicate a message is carrying internals out with it. Kept as
#: separate patterns rather than one alternation: a single combined pattern is
#: easy to get subtly unbalanced, and these are all cheap substring-ish shapes
#: that only need to fire on text that came from an exception.
_INTERNAL_DETAIL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # A Python traceback header.
    re.compile(r"Traceback \(most recent call last\)"),
    # A source-file reference from a traceback frame.
    re.compile(r'File "'),
    # A Windows absolute path, e.g. C:\Users\...
    re.compile(r"[A-Za-z]:\\"),
    # A UNC path, e.g. \\server\share.
    re.compile(r"\\\\[A-Za-z0-9_.-]+\\"),
    # Python package directories.
    re.compile(r"(?<![A-Za-z0-9_])(?:site-packages|dist-packages|__pycache__)"),
)

MAX_ERROR_MESSAGE_LENGTH = 300


def looks_internal(message: str) -> bool:
    """True when a message carries filesystem or implementation detail."""
    return any(pattern.search(message) for pattern in _INTERNAL_DETAIL_PATTERNS)


def _scrub_message(message: str, fallback: str = GENERIC_INTERNAL_MESSAGE) -> str:
    """Keep a message only if it is short and carries no internal detail."""
    if not isinstance(message, str):
        return fallback
    trimmed = message.strip()
    if not trimmed or len(trimmed) > MAX_ERROR_MESSAGE_LENGTH:
        return fallback
    if looks_internal(trimmed):
        return fallback
    return trimmed


def error_body(code: str, message: str, request_id: str | None) -> dict[str, Any]:
    """The one error shape every failure path returns.

    `detail` is retained alongside `error` because the existing 422 and 503
    contract (and the frontend, which reads `detail`) depends on it. `error` is
    the stable, correlation-carrying part.
    """
    safe = _scrub_message(message, GENERIC_INTERNAL_MESSAGE)
    return {
        "detail": safe,
        "error": {
            "code": code,
            "message": safe,
            "request_id": request_id or "",
        },
    }


# --------------------------------------------------------------------------- #
# Request identity
# --------------------------------------------------------------------------- #

#: The active request id, readable from anywhere in this request's context
#: without threading it through every call site. Used by the logging filter so
#: operational lines are correlatable without any health data being logged.
current_request_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "swasthyaguide_request_id", default="-"
)


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every log record on this logger.

    Adds one field and nothing else. It never sees a request body, so it cannot
    log one: request payloads and prediction results are never passed to a
    logger anywhere in this application.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id.get()  # type: ignore[attr-defined]
        return True


def attach_request_id_filter() -> None:
    """Idempotently install the filter on the application logger hierarchy."""
    target = logging.getLogger("swasthyguide")
    for handlerless in (target, logger):
        if not any(isinstance(f, RequestIdFilter) for f in handlerless.filters):
            handlerless.addFilter(RequestIdFilter())


def new_request_id() -> str:
    return uuid.uuid4().hex


def validate_request_id(candidate: str | None) -> str | None:
    """Accept an inbound correlation id only if it is a safe short token."""
    if not candidate or len(candidate) > MAX_INBOUND_REQUEST_ID_LENGTH:
        return None
    return candidate if VALID_REQUEST_ID.match(candidate) else None


def _header_value(scope: Scope, name: bytes) -> str | None:
    for key, value in scope.get("headers") or ():
        if key == name:
            try:
                return value.decode("latin-1")
            except UnicodeDecodeError:  # pragma: no cover - latin-1 never fails
                return None
    return None


def request_id_for(scope: Scope) -> str:
    inbound = validate_request_id(_header_value(scope, REQUEST_ID_HEADER_LOWER))
    return inbound or new_request_id()


# --------------------------------------------------------------------------- #
# Client identity, and not trusting forwarded headers by default
# --------------------------------------------------------------------------- #

_IPV4 = re.compile(r"\A\d{1,3}(?:\.\d{1,3}){3}\Z")


def _looks_like_ip(value: str) -> bool:
    if _IPV4.match(value):
        return all(0 <= int(part) <= 255 for part in value.split("."))
    return ":" in value or bool(value) and value.count(":") == 2


def client_ip(scope: Scope, trusted_proxy_hops: int = 0) -> str:
    """The address the rate limiter should count against.

    Forwarded headers are attacker-controlled. By default (`trusted_proxy_hops`
    == 0) only the transport peer is used, so a caller cannot rotate its own
    identity by inventing `X-Forwarded-For`.

    When an operator declares N trusted proxies in front of the app, the Nth
    entry from the right of `X-Forwarded-For` is used. That is the only position
    a proxy we control actually appended, so everything to its left is
    attacker-supplied noise that we ignore. The reverse proxy must strip
    client-supplied `X-Forwarded-For` before setting its own; see
    PRODUCTION_SECURITY_CHECKLIST.md.
    """
    peer = scope.get("client")
    peer_host = peer[0] if peer else None

    if trusted_proxy_hops > 0:
        forwarded = _header_value(scope, b"x-forwarded-for")
        if forwarded:
            hops = [part.strip() for part in forwarded.split(",") if part.strip()]
            if hops:
                index = max(0, len(hops) - trusted_proxy_hops)
                candidate = hops[index]
                if _looks_like_ip(candidate):
                    return candidate

    return peer_host or "unknown"


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: int


class RateLimiter:
    """The rate limiter contract.

    Two methods, no framework types. A Redis implementation satisfies this
    interface by swapping the storage in `check` and discarding `reset`; no other
    call site has to change, and Redis stays out of `requirements.txt` because
    the shipped deployment is single-instance.

    !!! warning
        An in-memory limiter is per-process. It is correct for a single
        instance and WRONG for a scaled deployment: N instances behind a load
        balancer give a caller N times the configured limit. Multi-instance
        production MUST move this state to a shared store.
    """

    def check(self, key: str, limit: int, window_seconds: float) -> RateLimitDecision:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryRateLimiter(RateLimiter):
    """Fixed-window counter, per process.

    Bounded on purpose: `max_keys` caps how many distinct client identities are
    tracked, so an attacker rotating spoofed client addresses cannot grow the
    process heap without limit. When the cap is hit the oldest half is evicted,
    which can only ever *under*-count for those identities and never lets the
    table grow unbounded.
    """

    def __init__(self, window_seconds: float = 60.0, max_keys: int = 10_000) -> None:
        self.window_seconds = float(window_seconds)
        self.max_keys = max_keys
        self._buckets: "OrderedDict[str, deque[float]]" = OrderedDict()

    def check(self, key: str, limit: int, window_seconds: float | None = None) -> RateLimitDecision:
        window = self.window_seconds if window_seconds is None else float(window_seconds)
        if limit <= 0:
            return RateLimitDecision(allowed=False, limit=limit, remaining=0,
                                     retry_after=max(1, int(window)))

        now = time.monotonic()
        bucket = self._buckets.get(key)
        if bucket is None:
            if len(self._buckets) >= self.max_keys:
                for _ in range(max(1, self.max_keys // 2)):
                    self._buckets.popitem(last=False)
            bucket = deque()
            self._buckets[key] = bucket
        else:
            self._buckets.move_to_end(key)

        cutoff = now - window
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= limit:
            retry_after = max(1, int(window - (now - bucket[0])))
            return RateLimitDecision(False, limit, 0, retry_after)

        bucket.append(now)
        return RateLimitDecision(True, limit, max(0, limit - len(bucket)), 0)

    def reset(self) -> None:
        self._buckets.clear()


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #

def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(raw: str | None, default: int, *, minimum: int = 1) -> int:
    if raw is None or not raw.strip():
        return default
    try:
        parsed = int(raw.strip())
    except ValueError:
        logger.warning("Ignoring non-numeric security setting %r; using %s", raw, default)
        return default
    if parsed < minimum:
        logger.warning("Security setting %r below minimum %s; using %s", raw, minimum, default)
        return default
    return parsed


def _csv(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return ()
    return tuple(part.strip() for part in raw.split(",") if part.strip())


#: Local development origins. The Vite dev server and the API itself; the
#: frontend normally calls same-origin through the /api proxy, but these keep a
#: direct cross-origin call working while developing.
DEFAULT_DEVELOPMENT_ORIGINS: tuple[str, ...] = (
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
)


@dataclass(frozen=True)
class SecuritySettings:
    """Every runtime security knob, resolved once and passed explicitly.

    Nothing here reads `os.environ` at call time, so a test can construct an
    exact configuration instead of mutating global state.
    """

    environment: str = DEVELOPMENT
    docs_enabled: bool = True
    cors_origins: tuple[str, ...] = DEFAULT_DEVELOPMENT_ORIGINS
    cors_allow_credentials: bool = False
    hsts_enabled: bool = False
    content_security_policy: str = DEFAULT_CSP
    permissions_policy: str = DEFAULT_PERMISSIONS_POLICY
    allowed_hosts: tuple[str, ...] = ()
    trusted_proxy_hops: int = 0
    max_request_size_bytes: int = 1_048_576
    rate_limit_enabled: bool = True
    rate_limits: Mapping[str, int] = field(default_factory=dict)
    expose_build_environment: bool = False
    trusted_proxy_hops_declared: bool = False
    allowed_hosts_declared: bool = False

    # -- derived ----------------------------------------------------------- #

    @property
    def is_production(self) -> bool:
        return self.environment == PRODUCTION

    def limit_for_path(self, path: str) -> int:
        """Per-endpoint limit, falling back to the site-wide default.

        `/health` is deliberately generous: a monitoring agent polling every few
        seconds must never be mistaken for abuse.
        """
        normalised = normalise_path(path)
        if normalised in self.rate_limits:
            return self.rate_limits[normalised]
        if normalised.startswith("/api/"):
            tail = normalised[len("/api"):]
            if tail in self.rate_limits:
                return self.rate_limits[tail]
        return self.rate_limits.get("*", 0)

    # -- construction ------------------------------------------------------ #

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "SecuritySettings":
        env = os.environ if environ is None else environ

        environment = (env.get(ENV_APP_ENV) or DEVELOPMENT).strip().lower() or DEVELOPMENT
        production = environment == PRODUCTION

        # HTTPS is the reason HSTS exists. Default it on for production and leave
        # it off for local HTTP development, so a developer on http://localhost
        # is never locked out of their own app.
        force_https = _truthy(env.get(ENV_FORCE_HTTPS), default=production)
        hsts_enabled = force_https or production

        # Documentation: off by default in production, on by default elsewhere,
        # and explicitly overridable in either direction.
        raw_docs = env.get(ENV_ENABLE_DOCS)
        if raw_docs is None:
            raw_docs = env.get(ENV_ENABLE_DOCS_LEGACY)
        docs_enabled = _truthy(raw_docs, default=not production)

        # CORS. Explicit origins only; there is deliberately no code path that
        # produces a wildcard.
        raw_origins = env.get(ENV_CORS_ORIGINS)
        if raw_origins is None:
            cors_origins = () if production else DEFAULT_DEVELOPMENT_ORIGINS
        else:
            cors_origins = _csv(raw_origins)
        cors_credentials = _truthy(env.get(ENV_CORS_CREDENTIALS), default=False)
        if cors_credentials and not cors_origins:
            raise ValueError(
                f"{ENV_CORS_CREDENTIALS} requires explicit {ENV_CORS_ORIGINS}; "
                "refusing to combine credentials with an origin allow-list that "
                "would otherwise fall back to a wildcard."
            )

        allowed_hosts = _csv(env.get(ENV_ALLOWED_HOSTS))
        trusted_proxy_hops = _positive_int(
            env.get(ENV_TRUSTED_PROXY_HOPS), 0, minimum=0
        )
        if trusted_proxy_hops < 0:  # pragma: no cover - guarded by minimum
            trusted_proxy_hops = 0

        return cls(
            environment=environment,
            docs_enabled=docs_enabled,
            cors_origins=cors_origins,
            cors_allow_credentials=cors_credentials,
            hsts_enabled=hsts_enabled,
            content_security_policy=(env.get(ENV_CSP) or DEFAULT_CSP).strip() or DEFAULT_CSP,
            allowed_hosts=allowed_hosts,
            trusted_proxy_hops=trusted_proxy_hops,
            max_request_size_bytes=_positive_int(
                env.get(ENV_MAX_BODY_BYTES), 1_048_576, minimum=1
            ),
            rate_limit_enabled=_truthy(env.get(ENV_RATE_LIMIT_ENABLED), default=True),
            expose_build_environment=_truthy(env.get(ENV_EXPOSE_BUILD_ENV), default=False),
            rate_limits={
                "*": _positive_int(env.get(ENV_RATE_LIMIT_DEFAULT), 60, minimum=0),
                "/predict": _positive_int(env.get(ENV_RATE_LIMIT_PREDICT), 20, minimum=0),
                "/health": _positive_int(env.get(ENV_RATE_LIMIT_HEALTH), 600, minimum=0),
            },
            allowed_hosts_declared=env.get(ENV_ALLOWED_HOSTS) is not None,
            trusted_proxy_hops_declared=env.get(ENV_TRUSTED_PROXY_HOPS) is not None,
        )


def normalise_path(path: str) -> str:
    """Trim a trailing slash so `/predict/` and `/predict` share one bucket."""
    if len(path) > 1 and path.endswith("/"):
        return path.rstrip("/") or "/"
    return path


# --------------------------------------------------------------------------- #
# ASGI helpers
# --------------------------------------------------------------------------- #

async def send_json(
    send: Send, status_code: int, payload: Mapping[str, Any], headers: list[tuple[bytes, bytes]]
) -> None:
    """Emit a response without depending on a Request object.

    Middleware can fail before routing, so error responses are built here rather
    than through a FastAPI exception handler.
    """
    body = JSONResponse(content=dict(payload), status_code=status_code)
    raw = body.body
    start_headers = list(headers)
    start_headers.append((b"content-type", b"application/json"))
    start_headers.append((b"content-length", str(len(raw)).encode("latin-1")))
    await send({"type": "http.response.start", "status": status_code, "headers": start_headers})
    await send({"type": "http.response.body", "body": raw})


def _header_list(*pairs: tuple[str, str]) -> list[tuple[bytes, bytes]]:
    return [(name.lower().encode("latin-1"), value.encode("latin-1")) for name, value in pairs]


def _append_header(headers: list[tuple[bytes, bytes]], name: bytes, value: bytes) -> None:
    """Add a header unless one with that name is already present.

    Prevents the duplicate-value response headers (`X-Request-ID: a, a`) that
    appear when two middlewares both contribute the same field.
    """
    if any(existing == name for existing, _ in headers):
        return
    headers.append((name, value))


# --------------------------------------------------------------------------- #
# 1 + 5. Request id and security headers
# --------------------------------------------------------------------------- #

class RequestIdMiddleware:
    """Assign, propagate and log a correlation id for every request.

    Outermost middleware, so the id exists before anything else can fail and is
    therefore present on every response including 413, 429 and 500. An id
    supplied by the caller is used only if it passes `validate_request_id`;
    otherwise a fresh one is generated and the rejection is logged.

    The access line records method, path, status, duration and the id. It never
    records a query string, a request body, or a response body, so no health
    input and no risk result reaches the log.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = request_id_for(scope)
        inbound = _header_value(scope, REQUEST_ID_HEADER_LOWER)
        if inbound and request_id != inbound:
            logger.warning("Discarded malformed inbound %s", REQUEST_ID_HEADER)

        scope.setdefault("state", {})["request_id"] = request_id
        token = current_request_id.set(request_id)
        started = time.monotonic()
        status_code = {"value": status.HTTP_500_INTERNAL_SERVER_ERROR}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_code["value"] = message["status"]
                headers = list(message.get("headers") or ())
                _append_header(
                    headers, REQUEST_ID_HEADER_LOWER, request_id.encode("latin-1")
                )
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # Log before resetting: the filter reads the id from the contextvar,
            # so resetting first would strip it from this very line.
            elapsed_ms = (time.monotonic() - started) * 1000.0
            try:
                logger.info(
                    "%s %s -> %s in %.1fms client=%s",
                    scope.get("method", "?"),
                    # Path only. The query string is dropped on purpose.
                    scope.get("path", "?"),
                    status_code["value"],
                    elapsed_ms,
                    client_ip(scope),
                )
            except Exception:  # pragma: no cover - logging must never break a response
                pass
            current_request_id.reset(token)


class SecurityHeadersMiddleware:
    """Attach the response security headers to every response.

    Outermost of the content-handling middlewares, so headers are present on
    responses produced by any of them, including CORS preflight rejections and
    rate-limit refusals. HSTS is emitted only when `hsts_enabled`, which is
    false for local HTTP development by design.
    """

    def __init__(self, app: ASGIApp, settings: SecuritySettings) -> None:
        self.app = app
        self.settings = settings

    def _headers(self) -> list[tuple[bytes, bytes]]:
        settings = self.settings
        pairs = [
            ("X-Content-Type-Options", "nosniff"),
            ("X-Frame-Options", "DENY"),
            ("Referrer-Policy", "no-referrer"),
            ("Permissions-Policy", settings.permissions_policy),
            ("Content-Security-Policy", settings.content_security_policy),
            ("Cross-Origin-Opener-Policy", "same-origin"),
            ("Cross-Origin-Resource-Policy", "same-origin"),
            ("X-DNS-Prefetch-Control", "off"),
            ("Cache-Control", API_CACHE_CONTROL),
            ("Pragma", "no-cache"),
        ]
        if settings.hsts_enabled:
            pairs.append(("Strict-Transport-Security", HSTS_VALUE))
        return _header_list(*pairs)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        headers = self._headers()

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                existing = list(message.get("headers") or ())
                present = {name for name, _ in existing}
                for name, value in headers:
                    # Never duplicate a header the app chose deliberately.
                    if name not in present:
                        existing.append((name, value))
                message = {**message, "headers": existing}
            await send(message)

        await self.app(scope, receive, send_wrapper)


# --------------------------------------------------------------------------- #
# 3. Rate limiting
# --------------------------------------------------------------------------- #

class RateLimitMiddleware:
    """Per-client, per-endpoint request budget.

    OPTIONS preflight is exempt: a browser sends one before every cross-origin
    POST, so counting it would halve every client's real budget and would make
    the limit depend on browser behaviour rather than on caller behaviour.
    """

    def __init__(self, app: ASGIApp, settings: SecuritySettings, limiter: RateLimiter) -> None:
        self.app = app
        self.settings = settings
        self.limiter = limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        settings = self.settings
        if (
            scope.get("type") != "http"
            or not settings.rate_limit_enabled
            or scope.get("method") == "OPTIONS"
        ):
            await self.app(scope, receive, send)
            return

        limit = settings.limit_for_path(scope.get("path", ""))
        if limit <= 0:
            await self.app(scope, receive, send)
            return

        key = f"{client_ip(scope, settings.trusted_proxy_hops)} {normalise_path(scope.get('path', ''))}"
        decision = self.limiter.check(key, limit)

        if decision.allowed:
            async def send_wrapper(message: Message) -> None:
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers") or ())
                    headers.append(
                        (
                            b"x-ratelimit-limit",
                            str(decision.limit).encode("latin-1"),
                        )
                    )
                    headers.append(
                        (
                            b"x-ratelimit-remaining",
                            str(decision.remaining).encode("latin-1"),
                        )
                    )
                    message = {**message, "headers": headers}
                await send(message)

            await self.app(scope, receive, send_wrapper)
            return

        logger.warning(
            "Rate limit exceeded for %s on %s (limit %s/min)",
            key.rsplit(" ", 1)[-1],
            normalise_path(scope.get("path", "")),
            decision.limit,
        )
        request_id = scope.get("state", {}).get("request_id", "")
        await send_json(
            send,
            status.HTTP_429_TOO_MANY_REQUESTS,
            error_body(CODE_RATE_LIMITED, GENERIC_RATE_LIMIT_MESSAGE, request_id),
            _header_list(("Retry-After", str(decision.retry_after))),
        )


# --------------------------------------------------------------------------- #
# 4. Request size limit
# --------------------------------------------------------------------------- #

class _RequestBodyTooLarge(Exception):
    """Internal signal that the body budget was exhausted mid-stream."""


class BodySizeLimitMiddleware:
    """Reject an oversized body before any prediction work is attempted.

    `Content-Length` is checked first, which settles the ordinary case without
    reading a byte. A chunked body with no declared length is caught by wrapping
    `receive` and failing the request as soon as the budget is passed, so an
    undeclared large body cannot be streamed in under the limit.
    """

    def __init__(self, app: ASGIApp, settings: SecuritySettings) -> None:
        self.app = app
        self.settings = settings

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        maximum = self.settings.max_request_size_bytes
        if scope.get("type") != "http" or maximum <= 0 or scope.get("method") == "GET":
            await self.app(scope, receive, send)
            return

        declared = _header_value(scope, b"content-length")
        if declared is not None:
            try:
                if int(declared) > maximum:
                    await self._reject(scope, send, declared_length=declared)
                    return
            except ValueError:
                # An unparseable Content-Length is handled by the ASGI server.
                pass

        received = {"count": 0}

        async def limited_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                received["count"] += len(message.get("body") or b"")
                if received["count"] > maximum:
                    raise _RequestBodyTooLarge
            return message

        started = False

        async def send_wrapper(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, send_wrapper)
        except _RequestBodyTooLarge:
            if started:  # pragma: no cover - nothing has responded yet in practice
                raise
            await self._reject(scope, send)

    async def _reject(
        self, scope: Scope, send: Send, declared_length: str | None = None
    ) -> None:
        logger.warning(
            "Rejected oversized request body (declared=%s, limit=%s bytes)",
            declared_length or "chunked",
            self.settings.max_request_size_bytes,
        )
        request_id = scope.get("state", {}).get("request_id", "")
        await send_json(
            send,
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            error_body(CODE_PAYLOAD_TOO_LARGE, GENERIC_PAYLOAD_MESSAGE, request_id),
            _header_list(),
        )


# --------------------------------------------------------------------------- #
# 6. Centralised safe error handling
# --------------------------------------------------------------------------- #

def _request_id_of(request: Request) -> str:
    state = getattr(request, "state", None)
    return getattr(state, "request_id", "") or ""


class SafeErrorMiddleware:
    """Last-resort net: turn any unhandled exception into a safe 500.

    Sits inside Starlette's own error middleware, so it handles everything the
    routes and the registered handlers did not, and it does not re-raise. That
    keeps a traceback on the operator's console while guaranteeing the client
    receives only the safe envelope plus the correlation id.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        request_id = scope.get("state", {}).get("request_id", "")

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers") or ())
                if request_id:
                    _append_header(
                        headers, REQUEST_ID_HEADER_LOWER, request_id.encode("latin-1")
                    )
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "Unhandled error while serving %s %s", scope.get("method", "?"),
                scope.get("path", "?"),
            )
            await send_json(
                send,
                status.HTTP_500_INTERNAL_SERVER_ERROR,
                error_body(CODE_INTERNAL, GENERIC_INTERNAL_MESSAGE, request_id),
                _header_list(("Cache-Control", API_CACHE_CONTROL)),
            )


def safe_validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """422 that keeps the field and the reason, and drops everything internal.

    FastAPI's default body also echoes `input` (the caller's own health values
    straight back into the response and into any intermediary log) and carries a
    `ctx` that can embed exception objects. `loc` and `msg` are kept because the
    frontend maps them onto form fields, which is the whole point of a 422.
    """
    issues = []
    errors = getattr(exc, "errors", None)
    for error in (errors() if callable(errors) else []) or []:
        location = [str(part) for part in error.get("loc", ())]
        issues.append(
            {
                "type": str(error.get("type", "value_error")),
                "loc": location,
                "msg": _scrub_message(
                    str(error.get("msg", "")), "Input is not valid for this field."
                ),
            }
        )
    request_id = _request_id_of(request)
    body = error_body(CODE_VALIDATION, "Some entries need attention before this can run.", request_id)
    body["detail"] = issues
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=body,
        headers={"Cache-Control": API_CACHE_CONTROL},
    )


def safe_http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Preserve deliberate status/detail pairs; scrub anything unsafe.

    The 503 the model-unavailable path raises carries a constant message the
    frontend already relies on, so it passes through byte-identical.
    """
    status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)
    detail = getattr(exc, "detail", None)
    code = getattr(exc, "error_code", None) or (
        CODE_MODEL_UNAVAILABLE if status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        else CODE_HTTP_ERROR
    )
    request_id = _request_id_of(request)
    body = error_body(code, detail if isinstance(detail, str) else GENERIC_HTTP_MESSAGE, request_id)
    if isinstance(detail, (int, float, bool)):
        body["detail"] = detail
    headers = dict(getattr(exc, "headers", None) or {})
    headers.setdefault("Cache-Control", API_CACHE_CONTROL)
    return JSONResponse(status_code=status_code, content=body, headers=headers)


def safe_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Backstop registered on Starlette's error middleware.

    `SafeErrorMiddleware` normally wins, because it sits inside it. This exists
    so an error raised by another middleware still produces the same safe shape.
    """
    logger.exception("Unhandled error outside the request pipeline")
    request_id = _request_id_of(request)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_body(CODE_INTERNAL, GENERIC_INTERNAL_MESSAGE, request_id),
        headers={"Cache-Control": API_CACHE_CONTROL},
    )


# --------------------------------------------------------------------------- #
# Installation
# --------------------------------------------------------------------------- #

def register_exception_handlers(application: FastAPI) -> None:
    from fastapi.exceptions import RequestValidationError

    application.add_exception_handler(RequestValidationError, safe_validation_exception_handler)
    application.add_exception_handler(StarletteHTTPException, safe_http_exception_handler)
    application.add_exception_handler(Exception, safe_unhandled_exception_handler)


def log_configuration(settings: SecuritySettings) -> None:
    """State, at startup, what is enforced and what is not.

    A control that is not configured must be visible in the operator log rather
    than silently absent.
    """
    logger.info(
        "Security: environment=%s docs=%s hsts=%s rate_limit=%s "
        "predict=%s/min health=%s/min default=%s/min max_body=%s bytes",
        settings.environment,
        "enabled" if settings.docs_enabled else "disabled",
        "enabled" if settings.hsts_enabled else "disabled",
        "on" if settings.rate_limit_enabled else "off",
        settings.limit_for_path("/predict"),
        settings.limit_for_path("/health"),
        settings.limit_for_path("/other"),
        settings.max_request_size_bytes,
    )
    if not settings.cors_origins:
        if settings.is_production:
            logger.warning(
                "%s is not set in production, so every cross-origin request is "
                "refused. Same-origin deployment needs no CORS; a split-origin "
                "deployment must set it explicitly.",
                ENV_CORS_ORIGINS,
            )
        else:
            logger.warning("%s resolved to no origins; cross-origin calls will be refused.",
                           ENV_CORS_ORIGINS)
    if not settings.allowed_hosts:
        logger.warning(
            "Host header validation is DISABLED: %s is not set. Set it to the "
            "public hostname(s), or enforce this at the reverse proxy.",
            ENV_ALLOWED_HOSTS,
        )
    if settings.trusted_proxy_hops == 0:
        logger.info(
            "Forwarded headers are not trusted (%s=0); the rate limiter counts "
            "the transport peer address.",
            ENV_TRUSTED_PROXY_HOPS,
        )


def install_security(
    application: FastAPI, settings: SecuritySettings, limiter: RateLimiter | None = None
) -> RateLimiter:
    """Wire every control onto an application.

    Starlette runs middleware in reverse registration order, so the calls below
    read bottom-up as the request path:

        request id -> body size -> rate limit -> security headers -> host -> cors -> routes

    Request id is outermost so every response carries the id. Security headers
    sit outside CORS so a preflight rejection is still labelled. CORS is
    innermost so it wraps the routes, which is what it needs to see.
    """
    attach_request_id_filter()
    limiter = limiter or InMemoryRateLimiter()

    register_exception_handlers(application)

    application.add_middleware(CORSMiddleware, **cors_options(settings))
    if settings.allowed_hosts and "*" not in settings.allowed_hosts:
        application.add_middleware(
            TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts)
        )
    application.add_middleware(SecurityHeadersMiddleware, settings=settings)
    if settings.rate_limit_enabled:
        application.add_middleware(RateLimitMiddleware, settings=settings, limiter=limiter)
    application.add_middleware(BodySizeLimitMiddleware, settings=settings)
    application.add_middleware(SafeErrorMiddleware)
    application.add_middleware(RequestIdMiddleware)

    application.state.security_settings = settings
    application.state.rate_limiter = limiter
    log_configuration(settings)
    return limiter


def cors_options(settings: SecuritySettings) -> dict[str, Any]:
    """Build the CORS middleware arguments.

    `allow_origins` is always an explicit tuple. Starlette treats an empty list
    as "no cross-origin access", which is the correct fail-closed result when
    production has not been configured, and there is no branch anywhere that can
    produce `*`.
    """
    if "*" in settings.cors_origins:
        raise ValueError(
            "Wildcard CORS is not supported. Configure explicit origins in "
            f"{ENV_CORS_ORIGINS}."
        )
    return {
        "allow_origins": list(settings.cors_origins),
        "allow_credentials": settings.cors_allow_credentials,
        "allow_methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Accept", "Content-Type", REQUEST_ID_HEADER],
        "expose_headers": [REQUEST_ID_HEADER, "Retry-After", "X-RateLimit-Limit",
                           "X-RateLimit-Remaining"],
        "max_age": 600,
    }


def build_build_environment_policy(environ: Mapping[str, str] | None = None) -> bool:
    """Whether `/model-info` may disclose the Stage 3 build environment block.

    Off by default. It carries an OS build string and exact dependency versions,
    which is a targeting aid for known CVEs and tells a caller nothing about the
    risk assessment itself. Operators who want it can opt in explicitly.
    """
    env = os.environ if environ is None else environ
    return _truthy(env.get(ENV_EXPOSE_BUILD_ENV), default=False)


__all__ = [
    "BodySizeLimitMiddleware",
    "InMemoryRateLimiter",
    "RateLimitMiddleware",
    "RateLimiter",
    "RateLimitDecision",
    "RequestIdMiddleware",
    "SafeErrorMiddleware",
    "SecurityHeadersMiddleware",
    "SecuritySettings",
    "build_build_environment_policy",
    "client_ip",
    "error_body",
    "install_security",
    "new_request_id",
    "normalise_path",
    "validate_request_id",
]