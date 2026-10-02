"""
Shared pytest configuration.

The rate limits the application enforces are deliberately tight in the shipped
configuration (`/predict` allows 20 requests a minute), and the functional suite
in `tests/test_api.py` issues well over a hundred predictions from a single
client address inside a few seconds. That suite is testing prediction
semantics, not abuse control, so the budget is raised here rather than
weakening the default.

These values must be set before `app.main` is imported, because the module-level
`app` resolves its configuration at import time. pytest loads `conftest.py`
ahead of any test module, which is what makes this work. `setdefault` is used so
an operator who deliberately exports a limit still gets it.

The rate limiter itself is covered properly, with real limits and real 429s, in
`tests/test_security.py`.
"""

from __future__ import annotations

import os

os.environ.setdefault("SWASTHYAGUIDE_RATE_LIMIT_ENABLED", "true")
os.environ.setdefault("RATE_LIMIT_PER_MINUTE", "100000")
os.environ.setdefault("PREDICT_RATE_LIMIT_PER_MINUTE", "100000")
os.environ.setdefault("HEALTH_RATE_LIMIT_PER_MINUTE", "100000")