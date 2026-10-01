# Contributing to SWASTHYAGUIDE 2.0

SWASTHYAGUIDE 2.0 is a research prototype for cardiovascular **risk
assessment**. It is not a medical device and not a diagnostic system.

## Development setup

Prerequisites: Python 3.11+ and Node.js 20+.

```bash
git clone https://github.com/Shivam-k3/SwasthyaGuide2.0.git
cd SwasthyaGuide2.0
pip install -r requirements.txt
cd web && npm install
```

Both servers are needed for a working app. Run them in separate terminals from
the repository root:

```bash
# Terminal 1 — FastAPI on :8000
uvicorn app.main:app --reload

# Terminal 2 — Vite on :5173
cd web && npm run dev
```

The frontend proxies `/api/*` to `127.0.0.1:8000`, so no CORS configuration is
required and the browser makes same-origin requests.

## Project structure

```
app/       FastAPI serving layer: routing, schemas, inference, input notes
src/       Frozen ML pipeline: preprocessing, training, validation, verification
data/      The UCI Heart Disease CSV. Read-only, checksum-verified on load
models/    Persisted pipelines and Stage 3 artifacts. Frozen
notebooks/ EDA, preprocessing and validation notebooks
tests/     Backend test suite (pytest)
web/       React + TypeScript + Vite frontend
```

The request path through the code:

```
Browser
  -> web/src/api/client.ts        typed fetch, error mapping
  -> app/main.py                  FastAPI route
  -> app/schemas.py               request validation (Pydantic)
  -> app/model_service.py         threshold guard, inference, response assembly
  -> models/tuned/*.joblib        persisted sklearn Pipeline
  -> result
```

## Running tests

```bash
python -m pytest -q                        # 123 backend tests
python -m src.verify_stage3                # 69 Stage 3 assertions

cd web
npm run typecheck                          # tsc, no emit
npm test                                   # 49 frontend tests
npm run build                              # production bundle
```

Run all of these before committing. `python -m src.verify_stage3` and
`python -m pytest` must both be run from the repository root so `app` and `src`
resolve as packages.

## Code style

- Python: follow the surrounding style. Type hints on public functions;
  docstrings that explain purpose, constraints and non-obvious decisions.
- TypeScript: strict mode, no `any`. Prefer named exports.
- Comments should explain **why**, not restate what the code does.
- Keep the API contract stable: field names, enum values and the response
  shape are consumed by the frontend and asserted in tests.

## ML pipeline rules

Stages 1–3 are frozen. Do not, without explicit authorisation:

- modify `data/` or its checksum;
- change the train/test split, the seed, or the stratification;
- change preprocessing methodology (sentinel rules, imputation, indicators,
  encoding, `ColumnTransformer` composition);
- retrain, replace, or regenerate anything under `models/`;
- change the operating threshold or its provenance;
- change Stage 3 evaluation methodology or audit results.

A frontend or API change is fine. A change that alters a prediction is not.
`tests/test_stage3_integrity.py` hashes the dataset and all 69 files under
`models/` against a recorded baseline; if that test fails, something frozen was
modified.

## Pull request guidelines

1. Run the full verification set above and paste the results.
2. Keep frontend and backend responsibilities separated: the frontend presents
   and validates, the backend scores. No inference logic in the browser.
3. Do not weaken a test to make it pass. Fix the cause.
4. Do not commit secrets, `.env` files, `node_modules/`, or build output.
5. Describe user-visible wording changes in the PR, since the medical-safety
   wording of this project is reviewed deliberately.
6. Note explicitly if a change touches `app/`, `src/`, `data/` or `models/`.
