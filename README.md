# SWASTHYAGUIDE 2.0

Machine-learning health-risk assessment research prototype.

> **Disclaimer — read this first.**
> SWASTHYAGUIDE 2.0 is a **research prototype** for health **risk assessment**.
> It is **not** a medical device, **not** a diagnostic system, and has not been
> validated or approved for clinical decision-making. Outputs must never be used
> to diagnose, treat, or rule out any condition. Always consult a qualified
> healthcare professional.

---

## Architecture

```
User
  ↓
web/  React + Vite (assessment form, review, result)
  ↓  /api/predict  (same-origin via the Vite dev proxy)
app/main.py  FastAPI route + lifespan
  ↓
app/schemas.py  request validation (Pydantic)
  ↓
app/model_service.py  threshold guard, inference, response assembly
  ↓
app/model_service.py:load_model  models/tuned/*.joblib
  ↓
persisted sklearn Pipeline  (imputation → encoding → scaling → estimator)
  ↓
result  score + classification + measured operating point + disclaimer
```

Three layers, with one direction of responsibility:

| Layer | Owns | Never does |
|-------|------|-------------|
| `web/` | presentation, form state, input validation, error copy | inference, thresholds, scoring |
| `app/` | validation, model loading, inference, response shape | presentation |
| `src/` | preprocessing, training, validation, verification | HTTP concerns |

The frontend holds no clinical logic. Every probability, threshold and
classification on screen is returned by the API.

### API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness and whether a model is loaded. `200` loaded, `503` not. |
| `GET` | `/model-info` | Model id, input schema, threshold provenance, calibration caveat. |
| `POST` | `/predict` | Score one patient. |

`/docs` serves the generated OpenAPI documentation.

The 13 request fields are `age`, `trestbps`, `chol`, `thalach`, `oldpeak`,
`ca`, `sex`, `cp`, `restecg`, `slope`, `thal`, `fbs`, `exang`. `age`, `sex` and
`cp` are required; the other ten accept `null`. Enum values are the fitted
encoder's own strings and must be sent exactly (`reversable defect`,
`non-anginal`, and so on).

---

## Project purpose

Build explainable ML modules that estimate **health risk** from routinely
collected patient attributes, and surface them through an API and a web
interface.

The **Heart Disease Risk** module is the **first** module, chosen because the
UCI Heart Disease dataset is small, well documented, and interpretable, which
makes it a good end-to-end template for the rest of the platform.

Current state: **Stage 5 complete — FastAPI serving layer plus the React web
interface.** The ML pipeline from Stages 1–3 is frozen and unchanged by both.
Security, privacy and medical-safety hardening is **not implemented** and is
planned as Stage 5.5.

---

## Dataset source

| Item | Value |
|------|-------|
| Name | UCI Heart Disease (processed Kaggle mirror) |
| Local path | `data/heart_disease_uci.csv` |
| Size | 79,346 bytes |
| SHA-256 | `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae` |
| Origin | Four UCI cohorts: Cleveland, Hungary, Switzerland, VA Long Beach |

`data/heart_disease_uci.csv` is **read-only input**. `src/preprocessing.py`
verifies the checksum on every load and raises if the file was altered, so an
accidental edit cannot pass silently.

---

## Dataset structure

920 rows x 16 columns, 1,759 missing cells in total.

| # | Column | dtype | Missing | Role | Description |
|---|--------|-------|---------|------|-------------|
| 1 | `id` | int64 | 0 | metadata — **excluded** | Row counter, 1..920 |
| 2 | `age` | int64 | 0 | numeric feature | Age in years (28–77) |
| 3 | `sex` | object | 0 | categorical | Male 726 / Female 194 |
| 4 | `dataset` | object | 0 | metadata — **excluded** | Source cohort (4 levels) |
| 5 | `cp` | object | 0 | categorical | Chest pain type (4 levels) |
| 6 | `trestbps` | float64 | 59 (6.41%) | numeric | Resting blood pressure, mm Hg |
| 7 | `chol` | float64 | 30 (3.26%) | numeric | Serum cholesterol, mg/dl |
| 8 | `fbs` | object | 90 (9.78%) | binary feature | Fasting blood sugar > 120 mg/dl (text `"True"`/`"False"`) |
| 9 | `restecg` | object | 2 (0.22%) | categorical | Resting ECG result (3 levels) |
| 10 | `thalch` | float64 | 55 (5.98%) | numeric | Max heart rate achieved, bpm |
| 11 | `exang` | object | 55 (5.98%) | binary feature | Exercise-induced angina (text `"True"`/`"False"`) |
| 12 | `oldpeak` | float64 | 62 (6.74%) | numeric | ST depression induced by exercise |
| 13 | `slope` | object | 309 (33.59%) | categorical | Peak exercise ST slope (3 levels) |
| 14 | `ca` | float64 | 611 (66.41%) | **ordinal** numeric | Major vessels with narrowing, 0–3 |
| 15 | `thal` | object | 486 (52.83%) | categorical | Thalassemia result (3 levels) |
| 16 | `num` | int64 | 0 | **target** | Ordinal 0–4 |

**13 model features** = all columns except `id`, `dataset`, `num`, `target`.

### Target definition

`num` is ordinal 0–4 as shipped. Binary rule applied in memory only:

```python
df["target"] = df["num"].gt(0).astype(int)   # 0 = no disease, 1 = disease
```

| Class | Rows | Share |
|-------|------|-------|
| `num == 0` → no disease | 411 | 44.67% |
| `num > 0` → disease | 509 | 55.33% |

---

## EDA findings (Stage 1)

Notebook: [`notebooks/01_heart_disease_eda.ipynb`](notebooks/01_heart_disease_eda.ipynb)

1. Target is `num`, confirmed present, ordinal 0–4 — not binary as shipped.
2. Binary collapse gives a mild imbalance: 44.7% / 55.3%.
3. **Prevalence differs wildly by cohort** — `num > 0` ranges 36.2% (Hungary) to
   93.5% (Switzerland).
4. **Missingness is cohort-structured** — `ca` 66.4%, `thal` 52.8%, `slope` 33.6%.
   Only 299 of 920 rows are complete.
5. **Implausible values** — `chol == 0` in 172 rows, `trestbps == 0` in 1 row,
   12 negative `oldpeak` values.
6. `fbs` and `exang` store strings `"True"`/`"False"`, not booleans.
7. `id` is a row counter, not a patient identifier.
8. No duplicate rows or ids; 2 duplicate feature profiles ignoring `id`/`dataset`.

### Missingness by cohort (%)

| Cohort | `trestbps` | `chol` | `fbs` | `thalch` | `exang` | `oldpeak` | `slope` | `ca` | `thal` |
|--------|-----------|--------|-------|----------|---------|----------|---------|-------|--------|
| Cleveland | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 1.6 | 1.0 |
| Hungary | 0.3 | 7.8 | 2.7 | 0.3 | 0.3 | 0.0 | 64.5 | 99.0 | 90.4 |
| Switzerland | 1.6 | 0.0 | 61.0 | 0.8 | 0.8 | 4.9 | 13.8 | 95.9 | 42.3 |
| VA Long Beach | 28.0 | 3.5 | 3.5 | 26.5 | 26.5 | 28.0 | 51.0 | 99.0 | 83.0 |

---

## Preprocessing decisions (Stage 2)

Implemented in [`src/preprocessing.py`](src/preprocessing.py). All of it lives
inside a `Pipeline` wrapping a `ColumnTransformer`, so nothing can leak from the
test set into training.

### 1. Invalid physiological values → NaN

| Column | Rule | Train rows | Reason |
|--------|------|-----------|--------|
| `chol` | `== 0` → NaN | 139 | Zero serum cholesterol is not measurable |
| `trestbps` | `== 0` → NaN | 1 | Zero resting BP is not measurable |
| `oldpeak` | `< 0` → NaN | 10 | ST depression is a magnitude, never negative |

Applied by `InvalidValueMasker` **inside** the pipeline, so inference data is
cleaned by the same rule. Implemented as a class rather than a closure so the
fitted pipeline stays picklable.

### 2. `ca`: ordinal numeric — inspected, not assumed

Evidence measured from the file:

* Values are exactly `{0, 1, 2, 3}` — integral, and in the original UCI spec `ca`
  is a **count** of narrowed vessels.
* Prevalence is **monotonic**: 26.5% → 68.7% → 80.5% → 85.0% disease.
* Spearman(ca, target) **rho = 0.484, p = 1.4e-19**.

**Decision: ordinal numeric**, because the count ordering matches the data.

**But median imputation alone would be biased.** 66.4% of `ca` is missing and 99%
of its non-missing values come from Cleveland alone, so the training median is
0.0 — every missing row gets "0 narrowed vessels", a healthy-looking value, even
though rows with missing `ca` are *more* likely to be diseased (59.7% vs 46.6%).
A `ca_missing` indicator is therefore mandatory, not decorative.

### 3. Boolean text → numeric

`fbs` and `exang` become floats 1.0/0.0 with NaN preserved, so an unrecorded
flag stays distinguishable from a recorded `False`. Unrecognised text becomes
NaN rather than being coerced.

### 4. Missing values — no rows dropped

Only 299/920 rows are complete, so deletion would discard two-thirds of the data.

* **Numeric** — median imputation fitted on training data only, plus a
  `missingindicator_<col>` column for each column with missing values.
* **Categorical** — explicit `Missing` level, then one-hot with
  `handle_unknown="ignore"` so unseen levels yield a zero row instead of raising.
* Row count is never reduced (asserted in notebook 02, section 8).

### 5. Final feature list

| # | Feature | Branch | Missing | Description |
|---|---------|--------|---------|-------------|
| 1 | `age` | numeric | 0.00% | Age in years |
| 2 | `trestbps` | numeric | 6.41% | Resting blood pressure |
| 3 | `chol` | numeric | 3.26% | Serum cholesterol |
| 4 | `thalch` | numeric | 5.98% | Max heart rate achieved |
| 5 | `oldpeak` | numeric | 6.74% | Exercise-induced ST depression |
| 6 | `ca` | numeric (ordinal) | 66.41% | Narrowed major vessels 0–3 |
| 7 | `sex` | categorical | 0.00% | Patient sex |
| 8 | `cp` | categorical | 0.00% | Chest pain type |
| 9 | `restecg` | categorical | 0.22% | Resting ECG result |
| 10 | `slope` | categorical | 33.59% | Peak exercise ST slope |
| 11 | `thal` | categorical | 52.83% | Thalassemia result |
| 12 | `fbs` | numeric (binary) | 9.78% | Fasting blood sugar > 120 |
| 13 | `exang` | numeric (binary) | 5.98% | Exercise-induced angina |

Never used as features: `id` (row counter), `dataset` (source metadata, used only
for splitting and reporting), `num` and `target` (the label).
`ColumnTransformer(remainder="drop")` guarantees they cannot reach a model.

The fitted pipeline expands 13 inputs into **33 columns**: 8 imputed features +
7 missingness indicators + 18 one-hot dummies (including 3 explicit `Missing`
levels).

---

## Train/test methodology

`train_test_split(test_size=0.2, random_state=42, stratify=cohort x label)`

Stratifying on the **joint** cohort × label key, not the label alone, is
deliberate: prevalence spans 36.2% to 93.5% across cohorts, so label-only
stratification could leave a cohort out of a partition or badly unbalanced
within it.

| Cohort | Total | Train | Test | Overall prev. | Train prev. | Test prev. |
|--------|-------|-------|------|---------------|-------------|------------|
| Cleveland | 304 | 243 | 61 | 0.4572 | 0.4568 | 0.4590 |
| Hungary | 293 | 235 | 58 | 0.3618 | 0.3617 | 0.3621 |
| Switzerland | 123 | 98 | 25 | 0.9350 | 0.9388 | 0.9200 |
| VA Long Beach | 200 | 160 | 40 | 0.7450 | 0.7438 | 0.7500 |

Train 736 rows (prevalence 0.5530), test 184 rows (prevalence 0.5543).

---

## Model results (Stage 2 — single holdout)

Un-tuned baselines at a fixed 0.5 decision threshold on the 184-row holdout. No
hyper-parameter search was performed in Stage 2, by design.

> These single-split numbers are superseded by the 25-fold results in Stage 3.
> They are kept because they are the only figures produced before any tuning, and
> they show how much a single split flatters a model compared with repeated CV.

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Accuracy |
|-------|---------|--------|--------|-----------|-----|----------|
| Logistic Regression | **0.8987** | **0.9038** | **0.9020** | 0.8364 | **0.8679** | **0.8478** |
| Random Forest | 0.8926 | 0.8832 | 0.8627 | **0.8544** | 0.8585 | 0.8424 |
| XGBoost | 0.8798 | 0.8743 | 0.8529 | 0.8286 | 0.8406 | 0.8207 |

ROC-AUC and PR-AUC are primary (threshold-free). Accuracy is **secondary**: the
positive class is the majority at 55.3%, so a constant "always disease"
predictor already scores 0.553 accuracy.

### Confusion matrices (test set, 184 rows, 102 positive)

| Model | TN | FP | FN | TP |
|-------|----|----|----|----|
| Logistic Regression | 64 | 18 | 10 | 92 |
| Random Forest | 67 | 15 | 14 | 88 |
| XGBoost | 64 | 18 | 15 | 87 |

### Reading the comparison

**No model is declared "best."** The three sit within a few points of each other
on ROC-AUC (0.88–0.90), and with 184 test rows those gaps are inside the noise.

* All three learn real signal — ROC-AUC well above the 0.5 chance line.
* Logistic Regression leads on both threshold-free metrics *and* is the
  simplest. On 736 training rows that is not a surprise.
* XGBoost ranks last despite being the most flexible model. Un-tuned boosting on
  a dataset this small is the regime where it overfits noise.
* All three share the same error profile: more false negatives than false
  positives, because the positive class is the majority.

---

## Cohort-wise results (Stage 2 — single holdout)

`reliable = False` marks a cohort with fewer than 50 test rows, or fewer than 10
positives or negatives — the figures exist but the uncertainty is too wide to act
on.

Note the flag flips for VA Long Beach between this table (40 holdout rows) and
the Stage 3 CV pool (160 rows, 41 negatives). The flag follows the sample, not
the cohort name.

| Model | Cohort | n | Pos | Prev. | ROC-AUC | PR-AUC | Recall | Prec. | F1 | Rel. |
|-------|--------|---|-----|-------|---------|--------|--------|-------|-----|------|
| Logistic Reg. | Cleveland | 61 | 28 | 0.459 | 0.9535 | 0.9362 | 0.9286 | 0.8125 | 0.8667 | yes |
| Logistic Reg. | Hungary | 58 | 21 | 0.362 | 0.8533 | 0.8275 | 0.7619 | 0.8000 | 0.7805 | yes |
| Logistic Reg. | Switzerland | 25 | 23 | 0.920 | 0.6522 | 0.9572 | 0.9565 | 0.9565 | 0.9565 | **no** |
| Logistic Reg. | VA Long Beach | 40 | 30 | 0.750 | 0.6867 | 0.8539 | 0.9333 | 0.8000 | 0.8615 | **no** |
| Random Forest | Cleveland | 61 | 28 | 0.459 | 0.9437 | 0.9025 | 0.9286 | 0.8667 | 0.8966 | yes |
| Random Forest | Hungary | 58 | 21 | 0.362 | 0.8610 | 0.8025 | 0.7143 | 0.7143 | 0.7143 | yes |
| Random Forest | Switzerland | 25 | 23 | 0.920 | 0.7609 | 0.9741 | 0.9130 | 0.9545 | 0.9333 | **no** |
| Random Forest | VA Long Beach | 40 | 30 | 0.750 | 0.7100 | 0.8293 | 0.8667 | 0.8667 | 0.8667 | **no** |
| XGBoost | Cleveland | 61 | 28 | 0.459 | 0.9437 | 0.9307 | 0.9643 | 0.9000 | 0.9310 | yes |
| XGBoost | Hungary | 58 | 21 | 0.362 | 0.8468 | 0.7947 | 0.6667 | 0.7368 | 0.7000 | yes |
| XGBoost | Switzerland | 25 | 23 | 0.920 | 0.5000 | 0.9212 | 0.8696 | 0.9091 | 0.8889 | **no** |
| XGBoost | VA Long Beach | 40 | 30 | 0.750 | 0.6900 | 0.8451 | 0.8667 | 0.7647 | 0.8125 | **no** |

The cohort picture matters more than the ranking:

* **Cleveland and Hungary** are the only cohorts with enough rows for a
  trustworthy figure (ROC-AUC 0.85–0.95).
* **Switzerland** (25 rows, 23 positive) and **VA Long Beach** (40 rows, 30
  positive) are unreliable. Base rates of 93.5% and 75% leave almost everything
  positive, so ROC-AUC is unstable — one XGBoost row lands at 0.50.
* The pooled 0.89 blends cohorts with 36% to 94% base rates. Useful as one
  number for this dataset, but it does not describe behaviour in any one cohort.

---

## Warnings and limitations

1. **920 rows, four cohorts.** Every metric carries wide uncertainty. After the
   paired per-fold test, two of the three model pairs are no longer separated on
   ROC-AUC at all, so the models are not ranked as better or worse.
2. **One holdout, evaluated once.** 25-fold cross-validation gives development
   estimates; the 184-row holdout is a single look and is now spent. Its numbers
   carry the uncertainty of one 184-row sample.
3. **Cohorts are not comparable.** Prevalence spans 36.2% to 93.9%, so pooled
   metrics average over clinically different populations. Switzerland has 98
   rows in the CV pool and is flagged unreliable at any model.
4. **`ca` correlates 0.516 with `num`.** Both describe vessel narrowing, so this
   is genuine information rather than a copied label — but part of the apparent
   strength comes from an angiographic measurement a screening setting would not
   have. Removing it costs 0.005-0.008 ROC-AUC (Stage 3 ablation).
5. **Search was partial for two models.** Random Forest scored 20 of 288 grid
   combinations and XGBoost 20 of 1296, sampled with replacement on the first
   CV repeat only. A different draw could select different parameters.
6. **Tuning deltas are biased upward.** The before/after comparison in Stage 3
   uses the same data that chose the hyper-parameters, so it cannot be read as
   evidence that tuning helped.
7. **`chol == 0` recoding is an interpretation.** Reading those 172 rows as "not
   recorded" is well supported (they cluster entirely in two cohorts) but it is a
   judgement call, and it is documented rather than silent.
8. **Random Forest probabilities are under-dispersed.** Calibration slope 1.33,
   expected calibration error 0.051 — roughly twice the others. It is *not*
   recalibrated, because recalibrating after seeing these results would be a
   modelling decision made on the evaluation data.
9. **No external validation.** Nothing has been tested outside this file.
10. **Research prototype for risk assessment.** Not a medical device, not a
   diagnostic system. These numbers must not be presented as clinical capability.

---

## Model selection and validation (Stage 3)

Notebook: [`notebooks/03_model_validation_explainability.ipynb`](notebooks/03_model_validation_explainability.ipynb)
Code: `src/validation.py`, `src/validate_stage3.py`, `src/explainability.py`,
`src/verify_stage3.py`

### 1. Protocol

```
920 rows ──► locked 80/20 split (cohort × label stratified, seed 42)
              ├── 736 train ──► 5-fold × 5-repeat StratifiedKFold
              │                  ├── un-tuned baselines
              │                  ├── bounded hyper-parameter search (first repeat only)
              │                  └── selected configurations
              │                        └── OOF predictions → threshold, calibration,
              │                            error analysis, case selection
              └── 184 holdout ──► touched ONCE, after all choices were frozen
```

Every number below the line "selected configurations" comes from out-of-fold
predictions on the 736 training rows. Each row's score is the **mean of its five
per-repeat predictions**; in every repeat the model scoring that row had never
seen it. The holdout is used exactly once, at the end, by
`src/validate_stage3.py` — never inside `src/validation.py`.

### 2. Selected configurations

| Model | Parameters | Search coverage |
|-------|-----------|-----------------|
| Logistic Regression | `C=0.3`, `l2`, `class_weight=None` | 16 of 16 (exhaustive) |
| Random Forest | 300 trees, `min_samples_leaf=2`, `max_depth=8`, `sqrt` features | 20 of 288 (**partial**) |
| XGBoost | 400 trees, `lr=0.03`, `max_depth=2`, `min_child_weight=5`, `subsample=1.0`, `reg_lambda=1.0` | 20 of 1296 (**partial**) |

Two budgets are deliberate and recorded in `models/stage3_report.json` rather
than hidden: the search runs on the **first repeat only** (5 folds, not 25), and
where the grid exceeds the budget sampling is **with replacement**, so some
combinations are never drawn. The artefacts state coverage per model.

### 3. Repeated cross-validation, 25 folds, threshold 0.5

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|-------|---------|--------|--------|-----------|-----|
| Logistic Regression | **0.9033** ±0.0197 | **0.9125** ±0.0202 | 0.8473 | **0.8378** | **0.8420** |
| Random Forest | 0.8970 ±0.0204 | 0.9008 ±0.0220 | **0.8572** | 0.8241 | 0.8398 |
| XGBoost | 0.8956 ±0.0194 | 0.9032 ±0.0219 | 0.8473 | 0.8334 | 0.8399 |

Mean ± standard deviation over the 25 folds.

**Comparing these means is not a test.** All three models ran on identical folds,
so fold difficulty is common to all three and largely cancels in a paired
difference. The per-fold rows are persisted to `models/cv_metrics_per_fold.csv`
and `models/cv_metrics_tuned_per_fold.csv` precisely so the comparison can be
made that way:

| Comparison | Un-tuned: mean gap ± paired SD | Selected: mean gap ± paired SD |
|------------|-------------------------------|-------------------------------|
| LR − RF | +0.0118 ± 0.0090 (22/25) | +0.0063 ± 0.0085 (20/25) — **not separated** |
| LR − XGB | +0.0307 ± 0.0136 (24/25) | +0.0077 ± 0.0074 (22/25) |
| RF − XGB | +0.0188 ± 0.0119 (25/25) | +0.0014 ± 0.0071 (15/25) — **not separated** |

The un-tuned baselines are separated, but **tuning erases most of that separation**: after tuning, two of the three pairs are indistinguishable on ROC-AUC and RF vs XGB is close to a coin flip (15 of 25 folds). On this dataset the three models are near-interchangeable on discrimination, and the choice between them rests on calibration and interpretability rather than on ROC-AUC.

Separable on ROC-AUC is not clinical superiority and does not transfer to a new
population. No p-value is reported: with 25 folds the binomial reference is coarse
and a significance test invites a ranking this dataset cannot support.

### 4. Threshold analysis

Full 5-point grid on selected-model OOF predictions:

| Threshold | LR recall / spec. | RF recall / spec. | XGB recall / spec. |
|-----------|------------------|------------------|-------------------|
| 0.30 | 0.936 / 0.626 | 0.963 / 0.575 | 0.939 / 0.626 |
| 0.40 | 0.897 / 0.736 | 0.929 / 0.672 | 0.892 / 0.708 |
| **0.50** | 0.845 / 0.799 | 0.862 / 0.781 | 0.845 / 0.809 |
| 0.60 | 0.789 / 0.848 | 0.769 / 0.854 | 0.799 / 0.842 |
| 0.70 | 0.727 / 0.900 | 0.659 / 0.903 | 0.720 / 0.888 |

Dropping to 0.30 buys about 9 extra sensitivity points on every model and costs
roughly 17 points of specificity, with F1 essentially flat (0.834–0.838 vs
0.842–0.846 at 0.50). **0.50 is kept** — no clinical threshold was specified, so
there is no evidence-based basis to trade specificity for sensitivity, and the
default is the one that required no post-hoc decision. ROC-AUC and PR-AUC are
unchanged across the grid, as they must be.

### 5. Calibration

| Model | Brier | Brier (constant baseline) | ECE | Intercept | Slope |
|-------|-------|--------------------------|-----|-----------|-------|
| Logistic Regression | **0.1226** | 0.2472 | **0.0239** | −0.0026 | 1.0133 |
| XGBoost | 0.1262 | 0.2472 | 0.0317 | 0.0029 | 1.0400 |
| Random Forest | 0.1285 | 0.2472 | 0.0510 | −0.0415 | 1.3318 |

All three beat the constant-prevalence baseline by roughly half. Logistic
Regression and XGBoost are close to the ideal slope of 1.0 with intercepts near
zero. **Random Forest is under-dispersed** (slope 1.33 — scores are not extreme
enough) and its expected calibration error is roughly twice the others, which is
visible in `models/calibration_curve.png`. It is not recalibrated: doing so after
seeing these results would be a modelling decision made on the evaluation data.

### 6. `ca` availability ablation

`ca` is 66.4% missing and almost entirely recorded for Cleveland. The same models
and folds were re-run without it:

| Model | ROC-AUC full → no-`ca` | Δ |
|-------|------------------------|---|
| Logistic Regression | 0.9033 → 0.8953 | −0.0080 |
| Random Forest | 0.8970 → 0.8923 | −0.0047 |
| XGBoost | 0.8956 → 0.8880 | −0.0076 |

Removing `ca` costs 0.005–0.008 ROC-AUC on every model, and costs more than
0.010 PR-AUC. This measures the price of the missing feature; it does not say
either variant is clinically preferable, and it does not resolve how a screening
setting would obtain angiographic counts.

### 7. Cross-validation by cohort (selected models, pooled OOF)

Reliability needs ≥50 rows, ≥10 positives and ≥10 negatives. The flag is computed
from the data and re-derived independently by the verification suite.

| Cohort | n | Prev. | LR ROC | RF ROC | XGB ROC | Reliable |
|--------|---|-------|--------|--------|--------|----------|
| Cleveland | 243 | 0.457 | 0.9011 | 0.8931 | 0.8969 | yes |
| Hungary | 235 | 0.362 | 0.9162 | 0.9136 | 0.9093 | yes |
| Switzerland | 98 | 0.939 | 0.7572 | 0.7011 | 0.6920 | **no** |
| VA Long Beach | 160 | 0.744 | 0.7379 | 0.7245 | 0.7161 | yes |

**Switzerland is the only unreliable slice** in the CV pool: 92 positives and 6
negatives makes ROC-AUC unstable no matter how many rows it has. VA Long Beach
*is* reliable here (160 rows, 41 negatives) — the flag differs by population, not
by cohort identity, which is why it is computed rather than listed.

### 8. Error analysis

Confusion counts at 0.5 on OOF predictions, broken down by cohort, by missingness
and by sex (`models/stage3_report.json`). Three patterns:

* Error rate is lowest where prevalence is highest (Switzerland 0.092) and
  highest for VA Long Beach (0.256, 160 rows, prevalence 0.744).
* Rows with missing values do **not** have a higher error rate (0.174) than
  complete rows (0.178) — median imputation plus indicators is not visibly
  costing accuracy here.
* Male rows have a higher error rate (0.191) than female rows (0.115), but male
  rows are also 3.6× more likely to be positive, so most of that gap is base rate
  rather than model behaviour.

These are associations on 920 rows, not causal findings.

### 9. Explainability

Coefficients, permutation importance and SHAP for all three models, plus
individual prediction explanations for 15 cases covering TP/FP/TN/FN and a
high-missingness row.

SHAP additivity is **verified numerically**, not assumed: base value + sum(SHAP
values) is checked against the model's own output on every explained row.

| Model | Verified against | Output scale | Max gap |
|-------|------------------|--------------|---------|
| Logistic Regression | `decision_function` | log-odds | 1.8e-15 |
| Random Forest | `predict_proba` | probability | 1.4e-15 |
| XGBoost | raw margin | log-odds | 4.7e-06 |

**Magnitudes are not comparable across model families** — one unit of XGBoost
SHAP is log-odds, one unit of Random Forest SHAP is probability. Only the ranking
of features within a model is meaningful.

Top features are consistent across all three families: `cp_asymptomatic`,
`exang`, `oldpeak`, `thalch`, `age`, then `sex` / `slope` / `thal`. `ca` appears
around 8th for the linear and tree models, which is what the ablation above would
predict given its 66.4% missingness — high individual predictive value where
present, little generalisable value because it is usually absent.

Feature attribution describes what the model used. It does not establish that a
feature causes disease, and it is not a clinical explanation.

### 10. Locked holdout — evaluated once

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Acc. | TN | FP | FN | TP |
|-------|---------|--------|--------|-----------|-----|------|----|----|----|----|
| Logistic Regression | 0.8998 | 0.9021 | 0.9020 | 0.8364 | 0.8679 | 0.8478 | 64 | 18 | 10 | 92 |
| Random Forest | 0.8974 | 0.8937 | 0.9118 | 0.8532 | 0.8815 | 0.8641 | 66 | 16 | 9 | 93 |
| XGBoost | 0.8972 | 0.9040 | 0.8922 | 0.8349 | 0.8626 | 0.8424 | 64 | 18 | 11 | 91 |

All three land within 0.008 of their own 25-fold CV ROC-AUC, and the three are
within 0.003 of each other on the holdout. **That convergence is reassuring about
the CV procedure, not a ranking** — 184 rows cannot separate models that 25 folds
also fail to separate.

### 11. Verification suite

`python -m src.verify_stage3` runs **69 assertions** covering reproducibility
(the raw CSV checksum), split leakage (the holdout is unreachable from inside
`src/validation.py`), preprocessing correctness (no dropped rows, no surviving
NaN, sentinel masking), artefact completeness, metric hygiene, the cohort
reliability rule, and dependency pinning.

```
69/69 checks passed
```

---

## Stage 4 — FastAPI serving layer

Code: `app/main.py`, `app/schemas.py`, `app/model_service.py`
Tests: `tests/test_api.py`, `tests/test_stage3_integrity.py` — **123 tests**

> **Disclaimer.** The API is a **research prototype** for health **risk
> assessment**. It is **not** a medical device, **not** a diagnostic system, and
> has not been validated for clinical decision-making. A response never states
> that a person does or does not have a condition. Always consult a qualified
> healthcare professional.

### 1. What Stage 4 is

A thin serving shell over the already-validated Stage 3 artifacts. **No new
modelling happened.** The persisted sklearn `Pipeline` is the source of truth,
so imputation, sentinel masking, encoding and scaling happen exactly as they
were fitted. Nothing in `src/`, `data/` or `models/` was modified.

### 2. How the model is loaded

`joblib.load` once, in a FastAPI `lifespan` handler, from
`models/tuned/{model}_pipeline.joblib`. The pipeline is held on `app.state`,
which is per-application, so two app instances in one process cannot see each
other's model. `fit()` is never called — a test monkeypatches `Pipeline.fit` to
raise and drives the whole API through it.

If loading fails the process still starts and still answers `/health`, because
a health endpoint that cannot respond is useless to whatever is monitoring it.
The failure surfaces as **503** with a single generic message; no filesystem
path, no traceback.

### 3. Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness and model availability. `200` when loaded, `503` when not. |
| `GET` | `/model-info` | Model id, estimator, provenance, input schema, threshold and its measured operating point, calibration caveat, environment. |
| `POST` | `/predict` | Score one patient. |

`/docs` serves the generated OpenAPI documentation.

### 4. Input schema

The 13 project features, no more and no fewer. Three rules decide every field,
and none of them is invented here:

* **Names and order** come from `src/preprocessing.py` (`FEATURE_COLUMNS`,
  `FEATURE_DOC`). The request frame is assembled in that exact order.
* **Category levels** match the fitted encoder's `categories_` minus the
  synthetic `Missing` level, as strict `Literal` enums. A misspelling such as
  `typcal angina` is a **422**, not a silently all-zero one-hot row. A test
  asserts these literals against the real artifact, so a future refit with new
  levels fails loudly.
* **Requiredness** comes from the source data's own missingness. `age`, `sex`
  and `cp` are complete in all 920 rows and are required; the other ten are
  optional, so absence stays representable and the Stage 2 `Missing` level and
  `missingindicator` columns do their job.

| Field | Type | Rule |
|-------|------|------|
| `age` | int | required, 1–120 |
| `sex` | enum | required: `Female`, `Male` |
| `cp` | enum | required: `asymptomatic`, `atypical angina`, `non-anginal`, `typical angina` |
| `trestbps`, `chol`, `thalch` | float or null | ≥ 0 |
| `oldpeak` | float or null | **unranged** — see below |
| `ca` | int or null | 0–3 (a count of narrowed major vessels) |
| `restecg`, `slope`, `thal` | enum or null | exactly the observed levels |
| `fbs`, `exang` | boolean or null | JSON `true`/`false` only |

Three deliberate choices:

* **`oldpeak` has no lower bound and `chol`/`trestbps` accept 0.** A zero
  cholesterol, a zero resting blood pressure and a negative ST depression are
  the project's documented invalid-sentinel convention — the source file uses
  `chol == 0` in 172 rows to mean "not recorded". The API accepts them, lets the
  persisted pipeline mask them exactly as it does in training, and **reports the
  interpretation** in `notes[]`. Rejecting them here would silently change the
  Stage 2 methodology.
* **`fbs` and `exang` are strict booleans.** Pydantic's lax mode accepts
  `"yes"`, `"on"` and `1` as `True`, which would flip a real risk input without
  anyone noticing. Only JSON `true`/`false` and `null` are accepted.
* **`thalch` and `thalach` are both accepted.** The dataset and the fitted
  pipeline spell the column `thalch`, a long-standing upstream UCI typo, and the
  pipeline name must not change. The correct clinical spelling is accepted as an
  alias for the same column. Supplying both is a 422.

### 5. Response

```json
{
  "model": "logistic_regression",
  "model_version": "stage3-2026-10-01T06:38:31+00:00/data-574f2fa2b430",
  "risk_probability": 0.558846,
  "prediction": "elevated_risk",
  "threshold": 0.4,
  "message": "Elevated predicted risk. Clinical evaluation may be appropriate. This is a research risk score, not a diagnosis.",
  "notes": [
    "chol=0 was treated as not recorded (zero serum cholesterol is not measurable) and imputed with a missingness indicator.",
    "Not supplied, so imputed from training data with a missingness indicator: restecg, slope, thal."
  ],
  "expected_operating_point": { "...": "see below" },
  "disclaimer": "Research prototype for health risk assessment. Not a medical device..."
}
```

A prediction is never phrased as a diagnosis. A low score says explicitly that
it **does not rule out** any condition.

### 6. Model selection

**Logistic Regression is the default.** On the 25-fold development estimate it
had the highest ROC-AUC and PR-AUC of the three and the best calibration, and it
is the simplest of the three to interpret.

> This is a **deployment choice** made on the project's own validation evidence
> and its interpretability requirements. It is **not** proof that Logistic
> Regression is universally better, and it does not transfer to an unseen
> population. Stage 3's paired per-fold comparison could not separate the three
> models on discrimination after tuning.

`SWASTHYAGUIDE_MODEL` selects any of `logistic_regression`, `random_forest` or
`xgboost`, all served from `models/tuned/`. `/model-info` carries the served
model's own calibration caveat, so Random Forest's documented under-dispersion
(slope 1.33, ECE 0.0510) travels with it rather than being discovered later.

### 7. Threshold, and the false-alarm / missed-alarm trade-off

SWASTHYAGUIDE's design rule is that **a false alarm is preferable to a missed
alarm**, so the served operating point favours recall over specificity.

Stage 3 kept **0.50** for research reporting because no clinical threshold had
been specified at that time. Stage 4 supplies that specification, which is what
makes moving the served threshold a deployment decision rather than a research
one. The move is bounded by evidence that already exists, and the threshold is
**per model**.

**Each model's default is its own measured operating point, not a number tuned
on one model and applied to another.** Stage 3 evaluated the full 0.3–0.7 grid
for all three models on the same 736-row out-of-fold pool, so every one has its
own validated recall/specificity trade-off at 0.40. Those are three independent
pieces of evidence, and the figures differ:

| Model | Default | Its own measured recall @ default | Its own measured specificity @ default |
|-------|---------|----------------------------------------|----------------------------------------------|
| Logistic Regression | 0.40 | 0.8968 | 0.7356 |
| Random Forest | 0.40 | 0.9287 | 0.6717 |
| XGBoost | 0.40 | 0.8919 | 0.7082 |

Borrowing Logistic Regression's tuned threshold for Random Forest would fabricate
validation that never happened. Stage 3 measured each model separately, and
`app/model_service.py` keeps them separate in `DEPLOYMENT_THRESHOLDS`.

For **Logistic Regression**, the default, moving 0.50 → 0.40 within the grid:

| Threshold | recall | specificity | precision | F1 | FN | FP |
|-----------|--------|-------------|-----------|----|----|----|
| 0.50 (Stage 3 reporting default) | 0.845 | 0.799 | 0.838 | 0.842 | 63 | 66 |
| **0.40 (Stage 4 deployment default)** | **0.897** | **0.736** | **0.808** | **0.850** | **42** | **87** |

That is ~5 more sensitivity points for ~6 points of specificity, with 21 fewer
missed cases out of 407 and F1 slightly improving. It is not free: 21 more
people are flagged who do not have the condition. **That trade-off is a clinical
decision, and the API does not make it silently** — it reports the measured cost
alongside the score. Random Forest and XGBoost carry the same trade-off on their
own numbers, which differ: at 0.40 RF reaches higher recall (0.929) at lower
specificity (0.672) than XGBoost (0.892 / 0.708).

Every response carries the **served model's own** measured behaviour at the
active threshold, read from `models/stage3_report.json`:

```json
"expected_operating_point": {
  "source": "Stage 3 out-of-fold predictions on the 736-row development pool. These describe the model's behaviour on that pool, not this patient. The locked holdout was not used.",
  "n_rows": 736, "n_positive": 407, "threshold": 0.4,
  "recall": 0.8968, "specificity": 0.7356, "precision": 0.8075, "f1": 0.8498,
  "false_positive_rate": 0.2644, "false_negative_rate": 0.1032,
  "roc_auc": 0.9037, "pr_auc": 0.9111,
  "tn": 242, "fp": 87, "fn": 42, "tp": 365
}
```

These numbers **differ by model** and are never shared across models:

| Model @ 0.40 | Recall | Specificity | FPR | FNR |
|--------------|--------|-------------|-----|-----|
| Logistic Regression | 0.8968 | 0.7356 | 0.2644 | 0.1032 |
| Random Forest | 0.9287 | 0.6717 | 0.3283 | 0.0713 |
| XGBoost | 0.8919 | 0.7082 | 0.2918 | 0.1081 |

**Three guards keep the threshold honest.** A model with no measured development
record cannot be served at all; a model with no entry in `DEPLOYMENT_THRESHOLDS`
has no validated operating point and is refused rather than given a borrowed or
invented one; and `SWASTHYAGUIDE_THRESHOLD` is accepted only if Stage 3 measured
that exact value **for the model actually being served**. Setting Logistic
Regression's 0.40 while serving Random Forest is therefore rejected at startup
rather than applied silently. Nothing is interpolated. The 184-row holdout is
never read.

The threshold is **server-side only**. `/predict` accepts no threshold field, so
a client cannot quietly disable the high-recall policy.

### 8. Stateless inference

A patient's score depends only on their own request. Verified by: bit-identical
repeats, the same patient scored alone and interleaved with unrelated patients,
reversed request order, and a direct check that batch position does not move the
pipeline's output. `predict()` mutates no fitted statistic, and the frozen
`models/` tree is byte-identical after the entire suite runs.

### 9. Running it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload              # development
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Run from the project root: uvicorn puts its `--app-dir` (default `.`) on
`sys.path`, so `app` and `src` resolve through ordinary package imports. No
`sys.path` bootstrap exists in `app/__init__.py`.

Optional, both server-side:

```bash
SWASTHYAGUIDE_MODEL=logistic_regression      # logistic_regression | random_forest | xgboost
SWASTHYAGUIDE_THRESHOLD=0.40                 # must be a measured grid point for the selected model
```

```bash
python -m pytest -q          # 123 tests, from the project root
```

`python -m pytest` is required rather than bare `pytest`, because the project
root has to be importable for `app` and `src` to resolve. No conftest
`sys.path` hack is used.

### 10. Stage 4 integrity

`tests/test_stage3_integrity.py` hashes `data/` and all 69 files under `models/`
and compares them against a baseline recorded **outside** the repository, before
any Stage 4 code existed. `tests/test_api.py` additionally re-runs the frozen
`python -m src.verify_stage3` suite as a subprocess and asserts 69/69.

Result: **69/69 Stage 3 checks still pass, and all 69 frozen artifacts are
byte-identical.** Stages 1–3 remain frozen. Any change to Stage 3 methodology
requires explicit authorisation and a fresh audit.

---

## Project structure

```
SWASTHYAGUIDE 2.0/
├── data/
│   └── heart_disease_uci.csv          # read-only, checksum-verified on load
├── notebooks/
│   ├── 01_heart_disease_eda.ipynb
│   ├── 02_preprocessing_and_baselines.ipynb
│   └── 03_model_validation_explainability.ipynb
├── models/
│   ├── logistic_regression_pipeline.joblib      # Stage 2 baselines
│   ├── random_forest_pipeline.joblib
│   ├── xgboost_pipeline.joblib
│   ├── tuned/                                   # Stage 3 selected pipelines
│   │   ├── logistic_regression_pipeline.joblib
│   │   ├── random_forest_pipeline.joblib
│   │   └── xgboost_pipeline.joblib
│   ├── explainability/                          # SHAP, importances, cases
│   ├── calibration_curve.png
│   ├── cv_metrics_overall.csv                   # un-tuned, mean ± SD
│   ├── cv_metrics_per_fold.csv                  # un-tuned, per fold
│   ├── cv_metrics_tuned_overall.csv             # selected, mean ± SD
│   ├── cv_metrics_tuned_per_fold.csv            # selected, per fold
│   ├── cv_metrics_by_cohort.csv / cv_metrics_tuned_by_cohort.csv
│   ├── threshold_analysis.csv
│   ├── calibration_metrics.csv / calibration_bins.csv
│   ├── feature_ablation_ca.csv / feature_ablation_ca_by_cohort.csv
│   ├── holdout_final_metrics.csv
│   ├── baseline_report.json / baseline_metrics_overall.csv
│   ├── baseline_metrics_by_cohort.csv
│   ├── split_summary.csv
│   └── stage3_report.json                       # consolidated Stage 3 record
├── src/
│   ├── preprocessing.py               # schema, target rule, pipeline
│   ├── train.py                       # split, fit, evaluate, persist
│   ├── evaluate.py                    # metrics + cohort reporting
│   ├── validation.py                  # CV, tuning, calibration, ablation
│   ├── explainability.py              # coefficients, SHAP, additivity
│   ├── validate_stage3.py             # Stage 3 driver, holdout last
│   ├── verify_stage3.py               # 69 assertions
│   └── __init__.py
├── app/                               # Stage 4 serving layer
│   ├── __init__.py
│   ├── main.py                        # ASGI app, lifespan, 3 endpoints
│   ├── model_service.py               # load, threshold guard, inference, metadata
│   ├── input_notes.py                 # how the pipeline will read each input
│   └── schemas.py                     # Pydantic request/response
├── web/                               # Stage 5 React + Vite frontend
│   ├── src/
│   │   ├── api/                       # typed client; strict request/response types
│   │   ├── components/assessment/     # form fields, two-layer field copy, review
│   │   ├── components/layout/         # header, footer, heart hero
│   │   ├── components/result/         # result presentation only
│   │   ├── components/ui/             # button, panel, section, error notice
│   │   ├── config/fields.tsx          # single field registry: API key, label, unit
│   │   ├── pages/                     # Home, Assessment, HowItWorks, About
│   │   ├── state/wizard.ts            # step state and request mapping
│   │   └── test/                      # Vitest suites
│   ├── public/                        # favicon, heart hero asset
│   ├── index.html
│   ├── package.json
│   └── vite.config.ts                 # dev proxy /api -> 127.0.0.1:8000
├── tests/
│   ├── test_api.py                    # 108 API tests
│   └── test_stage3_integrity.py       # 15 frozen-artifact tests
├── .gitignore
├── STAGE_3_AUDIT.md                   # read-only Stage 3 audit, 10/10 checks
├── CONTRIBUTING.md                    # setup, conventions, frozen-pipeline rules
├── requirements.txt
└── README.md
```

### Request path through the code

| File | Responsibility |
|------|----------------|
| `web/src/api/client.ts` | Typed `fetch`, status-to-message mapping, 422 field extraction |
| `web/src/state/wizard.ts` | Form state, serialisation to the API body, review formatting |
| `web/src/config/fields.tsx` | One field registry: API key, plain label, clinical term, unit, bounds |
| `app/main.py` | Routing and lifespan. No scoring logic |
| `app/schemas.py` | Request validation; builds the pipeline input frame |
| `app/input_notes.py` | Explains how the pipeline will interpret ambiguous inputs |
| `app/model_service.py` | Threshold guard, `joblib` load, inference, response assembly |

---

## Current development stage

**Stage 5 complete.**

| Stage | Status |
|-------|--------|
| 1. Dataset inspection and EDA | **Complete** |
| 2. Preprocessing pipeline + un-tuned baselines | **Complete** |
| 3. Hyper-parameter tuning, repeated CV and explainability | **Complete — frozen** |
| 4. FastAPI serving layer | **Complete** |
| 5. React web interface | **Complete** |
| 5.5. Security, privacy and medical-safety hardening | **Not implemented — planned** |

Stages 1–3 are frozen: the dataset, the train/test split, the three fitted
pipelines, the threshold analysis and the audit results are unchanged by Stages
4 and 5. The frontend is a presentation layer over the Stage 4 API and performs
no scoring of its own.

**Security, privacy and medical-safety hardening has not been started.** In
particular there is no authentication, no rate limiting, no security-header
configuration, no consent or retention mechanism, and no production transport
configuration. The application is a local research prototype and is not
configured for public deployment. That work is Stage 5.5 and has not begun.

---

## Reproducibility

- `RANDOM_SEED = 42`, fixed throughout.
- Exact dependency versions pinned in `requirements.txt` (14 packages, all pinned).
- Dataset SHA-256 checked on every load; recorded in the README, printed by all
  three notebooks, and stored in `models/baseline_report.json` and
  `models/stage3_report.json`.
- Raw CSV never modified; every transformation documented and reversible.
- Refitting produces bit-identical model coefficients (verified).
- Cross-validation on identical folds reproduces identical out-of-fold
  predictions, max difference `0.00e+00` (verified).
- SHAP additivity reproduced against each model's own output (verified).

```bash
pip install -r requirements.txt
python -m src.train            # Stage 2: split, fit, evaluate, write models/
python -m src.validate_stage3  # Stage 3: CV, tuning, explainability, holdout last
python -m src.verify_stage3    # 69 assertions
python -m pytest -q            # Stage 4: 123 API and integrity tests
uvicorn app.main:app --reload  # Stage 4: serve the API on :8000
jupyter lab notebooks/         # all three notebooks

cd web
npm install
npm run dev                    # Stage 5: frontend on :5173, /api proxied to :8000
npm run typecheck              # tsc -b --noEmit
npm test                       # 49 Vitest tests
npm run build                  # production bundle into web/dist
```

Run the backend and the frontend in separate terminals: the frontend expects the
API at `127.0.0.1:8000`.

Useful `src.train` flags: `--test-size`, `--seed`, `--threshold`.

`src.validate_stage3` writes every Stage 3 artefact under `models/`, including
`stage3_report.json`, and prints the full validation log. It reads the 184-row
holdout exactly once, after every modelling choice is frozen.

---

## What was implemented in Stage 3

For reference, the Stage 3 scope, all complete:

1. Repeated stratified cross-validation on the training split only, with per-fold
   rows persisted so model differences can be compared paired.
2. Bounded hyper-parameter search for all three models under one declared budget,
   with coverage recorded per model rather than implied.
3. Selected-configuration pipelines persisted to `models/tuned/`, plus
   `models/stage3_report.json` carrying cross-validated metrics, the OOF policy
   and the operating threshold.
4. Calibration assessment (Brier, ECE, slope, intercept, reliability curve),
   since risk *scores* are only meaningful if the probabilities can be trusted.
5. `ca` availability ablation, threshold grid, per-cohort CV, and error analysis.
6. SHAP and coefficient explainability with verified additivity.
7. A 69-assertion verification suite.

## What was implemented in Stage 4

1. A FastAPI application loading the persisted `models/tuned/` pipeline once at
   startup, never training and never reimplementing preprocessing.
2. Pydantic request/response schemas whose names, category levels and
   requiredness are all derived from existing Stage 1–3 constants rather than
   restated.
3. A high-recall deployment threshold, selected from Stage 3's measured grid and
   guarded so no unmeasured operating point can be served and no model's figures
   are reused for another.
4. Per-response reporting of the served model's measured recall, specificity,
   precision, F1, false-positive rate, false-negative rate, ROC-AUC and PR-AUC at
   the active threshold, so the false-alarm / missed-alarm cost is visible.
5. Clean 503 handling for missing, corrupt or unconfigured model artifacts, with
   no filesystem detail in any response.
6. A 123-test suite covering validation, statelessness, threshold provenance,
   medical-safety language, no-retraining, and byte-level integrity of all 69
   frozen Stage 1–3 artifacts.

## What was implemented in Stage 5

1. A React + TypeScript + Vite frontend in `web/` that consumes the Stage 4 API
   and performs no modelling of its own.
2. A four-page interface — Home, Assessment, How It Works, About — with one `h1`
   per page, a skip-to-content link, semantic form controls and visible focus
   states.
3. A three-step assessment (details → review → result) over the 13 API fields,
   with `Not available` preserved on every optional field so absence stays
   representable and nothing is silently defaulted.
4. A two-layer field copy system: plain-English labels and helper text as the
   primary layer, with the clinical variable name kept alongside it as secondary
   text for research and clinical readers.
5. Result presentation that never states or implies a diagnosis, shows the
   served model's own measured error rates at the active threshold, and renders
   the API's own disclaimer verbatim.
6. Typed error handling: `422` returns to the form with a field-level message,
   `503` shows a plain availability notice, and no traceback, filesystem path or
   environment value reaches the page.
7. A 49-test Vitest suite and a development `/api` proxy so the browser makes
   same-origin requests without backend CORS changes.

## What comes next

**Stage 5.5 — security, privacy and medical safety.** Not implemented. Planned
scope only:

* Transport and header hardening, CORS review, rate limiting, payload limits,
  dependency audit, secrets management, `/docs` and `/model-info` exposure
  review.
* Data minimisation, health-data logging review, retention and deletion policy,
  privacy notice, consent handling, third-party analytics review.
* Stronger non-diagnostic result wording, emergency symptom guidance, explicit
  model, population, generalisation and threshold-interpretation limitations.

Independently of any of that, the honest limitations remain: 920 rows, four
non-comparable cohorts, no external validation, no clinical validation, and
three models that repeated CV cannot separate on discrimination. Serving the API
and shipping a frontend change none of them.
