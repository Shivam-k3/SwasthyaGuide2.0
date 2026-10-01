"""
SWASTHYAGUIDE 2.0 — Heart Disease Risk module
Stage 4: FastAPI serving layer.

No sys.path manipulation here on purpose. `uvicorn` inserts its `--app-dir`
(default ".") into `sys.path` before importing the ASGI app, so running
`uvicorn app.main:app` from the project root makes both `app` and `src`
importable through ordinary package imports, and `python -m pytest` does the
same for the test suite. A bootstrap would only add a side effect to every
importer.

Research prototype for health RISK ASSESSMENT. Not a medical device, not a
diagnostic system.
"""
