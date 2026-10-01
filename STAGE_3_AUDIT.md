# Stage 3 Audit

**Audit date:** 2026-10-01
**Scope:** Pre-deployment verification of Stage 3 artifacts and methodology
**Result:** **10 / 10 audit checks passed**
**Status:** Stage 3 accepted. Stage 4 (FastAPI) and Stage 5 (Streamlit) remain **not started and not authorised**.

> **Disclaimer.** SWASTHYAGUIDE 2.0 is a **research prototype** for health **risk assessment**. It is **not** a medical device, **not** a diagnostic system, and has not been validated or approved for clinical decision-making. Outputs must never be used to diagnose, treat, or rule out any condition. Always consult a qualified healthcare professional.

This audit is read-only. It verified existing artifacts without changing methodology, without re-running the pipeline, and without overwriting any result. No file under `models/`, `data/`, `src/`, or `notebooks/` was modified during the audit, and `app/` remains empty.

---

## Audit result summary

| # | Check | Result | Key evidence |
|---|-------|--------|--------------|
| 1 | Persisted model files load | **PASS** | 6 / 6 pipelines load; all expose `predict_proba` |
| 2 | Single new patient row, no leakage | **PASS** | `(1, 13) → (1, 33) → (1, 2)`; order-invariant; statistics unchanged after `predict()` |
| 3 | Train/test split preserved | **PASS** | 736 / 184; disjoint; Stage 2 and Stage 3 entry points agree |
| 4 | Holdout unused for tuning / threshold / selection | **PASS** | No fold index reaches the holdout; no CV function receives test data |
| 5 | `ca` ablation persisted | **PASS** | 6 rows overall + 24 cohort rows; both variants for all 3 models |
| 6 | Calibration artifacts persisted | **PASS** | Brier / ECE / slope / intercept + bins + 97.8 KB reliability diagram |
| 7 | SHAP / explainability reproducible | **PASS** | Persisted vs recomputed global table: max diff `9.8e-17` |
| 8 | Verification suite | **PASS** | `69/69 checks passed` |
| 9 | Raw CSV SHA-256 unchanged | **PASS** | `574f2fa2…5a072ae`, 79,346 bytes |
| 10 | README accurately describes Stage 3 | **PASS** | 22 / 22 documentation assertions |

---

## 1. Completed experiments

### 1.1 Repeated cross-validation, 25 folds

Stratified 5-fold × 5-repeat cross-validation on the 736-row training portion, cohort × target strata, seed 42. All three models share byte-identical folds.

**Un-tuned baselines** (`models/cv_metrics_overall.csv`, mean ± SD over 25 folds):

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| Logistic Regression | 0.9020 ±0.0197 | 0.9103 ±0.0198 |
| Random Forest | 0.8902 ±0.0209 | 0.8909 ±0.0229 |
| XGBoost | 0.8714 ±0.0202 | 0.8697 ±0.0212 |

**Selected configurations** (`models/cv_metrics_tuned_overall.csv`):

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 |
|-------|---------|--------|--------|-----------|-----|
| Logistic Regression | **0.9033** ±0.0197 | **0.9125** ±0.0202 | 0.8473 | **0.8378** | **0.8420** |
| Random Forest | 0.8970 ±0.0204 | 0.9008 ±0.0220 | **0.8572** | 0.8241 | 0.8398 |
| XGBoost | 0.8956 ±0.0194 | 0.9032 ±0.0219 | 0.8473 | 0.8334 | 0.8399 |

### 1.2 Paired model comparison

Comparing one model's SD against another model's mean is **not** a valid test when all models share the same folds — fold difficulty is common to both sides and cancels. Per-fold rows are therefore persisted so comparisons can be paired.

| Pair | Un-tuned: mean gap ± paired SD | Selected: mean gap ± paired SD |
|------|-------------------------------|-------------------------------|
| LR − RF | +0.0118 ± 0.0090 (22/25) | +0.0063 ± 0.0085 (20/25) — **not separated** |
| LR − XGB | +0.0307 ± 0.0136 (24/25) | +0.0077 ± 0.0074 (22/25) |
| RF − XGB | +0.0188 ± 0.0119 (25/25) | +0.0014 ± 0.0071 (15/25) — **not separated** |

**Conclusion:** tuning erases most of the un-tuned separation. Two of three pairs are indistinguishable on ROC-AUC after selection, and RF vs XGB is close to a coin flip (15 of 25 folds). On this dataset the three models are near-interchangeable on **model discrimination**; the choice between them rests on calibration and interpretability.

Separable on ROC-AUC is **not** clinical superiority and does not transfer to a new population. No p-value is reported: with 25 folds the binomial reference is coarse, and a significance test would invite a ranking this dataset cannot support.

### 1.3 Bounded hyper-parameter search

| Model | Selected parameters | Coverage |
|-------|---------------------|----------|
| Logistic Regression | `C=0.3`, `l2`, `class_weight=None` | 16 of 16 (exhaustive) |
| Random Forest | 300 trees, `min_samples_leaf=2`, `max_depth=8`, `sqrt` | 20 of 288 (**partial**) |
| XGBoost | 400 trees, `lr=0.03`, `max_depth=2`, `min_child_weight=5` | 20 of 1296 (**partial**) |

Two budget decisions are declared rather than hidden: the search runs on the **first CV repeat only** (5 folds, not 25), and where the grid exceeds the budget, sampling is **with replacement**, so some combinations are never drawn. Coverage is recorded per model in `models/stage3_report.json`.

### 1.4 Threshold analysis

Full 5-point grid on selected-model out-of-fold predictions (`models/threshold_analysis.csv`):

| Threshold | LR recall / spec. | RF recall / spec. | XGB recall / spec. |
|-----------|------------------|------------------|-------------------|
| 0.30 | 0.936 / 0.626 | 0.963 / 0.575 | 0.939 / 0.626 |
| 0.40 | 0.897 / 0.736 | 0.929 / 0.672 | 0.892 / 0.708 |
| **0.50** | 0.845 / 0.799 | 0.862 / 0.781 | 0.845 / 0.809 |
| 0.60 | 0.789 / 0.848 | 0.769 / 0.854 | 0.799 / 0.842 |
| 0.70 | 0.727 / 0.900 | 0.659 / 0.903 | 0.720 / 0.888 |

**0.50 retained.** Dropping to 0.30 buys ~9 sensitivity points and costs ~17 specificity points with F1 essentially flat. No clinical threshold was specified, so there is no evidence-based basis for trading one for the other, and the default is the one that required no post-hoc decision. ROC-AUC and PR-AUC are unchanged across the grid, as they must be.

### 1.5 Calibration

| Model | Brier | Brier (constant) | ECE | Intercept | Slope |
|-------|-------|------------------|-----|-----------|-------|
| Logistic Regression | **0.1226** | 0.2472 | **0.0239** | −0.0026 | 1.0133 |
| XGBoost | 0.1262 | 0.2472 | 0.0317 | +0.0029 | 1.0400 |
| Random Forest | 0.1285 | 0.2472 | 0.0510 | −0.0415 | 1.3318 |

All three beat the constant-prevalence baseline by roughly half. Logistic Regression and XGBoost sit close to the ideal slope of 1.0 with intercepts near zero. **Random Forest is under-dispersed** (slope 1.33 — scores are not extreme enough) with roughly double the expected calibration error. It is **not** recalibrated: doing so after seeing these results would be a modelling decision made on the evaluation data.

### 1.6 `ca` availability ablation

| Model | ROC-AUC full → no-`ca` | Δ ROC-AUC |
|-------|------------------------|-----------|
| Logistic Regression | 0.9033 → 0.8953 | −0.0080 |
| Random Forest | 0.8970 → 0.8923 | −0.0047 |
| XGBoost | 0.8956 → 0.8880 | −0.0076 |

`ca` is 66.4% missing and 99% of its non-missing values come from Cleveland. Removing it costs 0.005–0.008 ROC-AUC and more than 0.010 PR-AUC on every model. This measures the **price of the missing feature**. It does not say either variant is clinically preferable, and it does not resolve how a screening setting would obtain angiographic counts.

### 1.7 Cross-validation by cohort

Reliability requires ≥50 rows, ≥10 positives and ≥10 negatives. The flag is computed from the data and independently re-derived by the verification suite.

| Cohort | n | Prev. | LR ROC | RF ROC | XGB ROC | Reliable |
|--------|---|-------|--------|--------|--------|----------|
| Cleveland | 243 | 0.457 | 0.9011 | 0.8931 | 0.8969 | yes |
| Hungary | 235 | 0.362 | 0.9162 | 0.9136 | 0.9093 | yes |
| Switzerland | 98 | 0.939 | 0.7572 | 0.7011 | 0.6920 | **no** |
| VA Long Beach | 160 | 0.744 | 0.7379 | 0.7245 | 0.7161 | yes |

Switzerland has 92 positives and 6 negatives, which makes ROC-AUC unstable regardless of row count. VA Long Beach **is** reliable in this 736-row pool but drops to 40 rows in the holdout — the flag follows the sample, not the cohort name, which is why it is computed rather than listed.

### 1.8 Error analysis

Confusion counts at 0.5 on out-of-fold predictions, broken down by cohort, missingness and sex (`models/stage3_report.json` → `error_analysis`).

* Error rate is lowest where prevalence is highest (Switzerland 0.092) and highest for VA Long Beach (0.256, n=160, prevalence 0.744).
* Rows with missing values do **not** show a higher error rate (0.174) than complete rows (0.178) — median imputation plus missingness indicators is not visibly costing accuracy.
* Male rows show a higher error rate (0.191) than female rows (0.115), but male rows are also ~3.6× more likely to be positive, so most of that gap is base rate rather than model behaviour.

These are associations on 920 rows, **not** causal findings.

### 1.9 Explainability

Coefficients, permutation importance and SHAP for all three models, plus 15 individual prediction explanations (3 models × TP / FP / TN / FN / high-missingness cases).

**SHAP additivity is verified numerically**, not assumed. The output scale is discovered by testing candidate model outputs rather than inferred from the model family:

| Model | Verified against | Scale | Max gap |
|-------|------------------|-------|---------|
| Logistic Regression | `decision_function` | log-odds | 1.8e-15 |
| Random Forest | `predict_proba` | probability | 1.4e-15 |
| XGBoost | raw margin | log-odds | 4.7e-06 |

**Magnitudes are not comparable across model families.** One unit of XGBoost SHAP is log-odds; one unit of Random Forest SHAP is probability. Only the within-model ranking is meaningful.

Top features are consistent across all three families: `cp_asymptomatic`, `exang`, `oldpeak`, `thalch`, `age`, then `sex` / `slope` / `thal`. `ca` appears around 8th for the linear and tree models, consistent with its high individual predictive value where present and low generalisable value because it is usually absent.

Feature attribution describes what the model used. It does **not** establish that a feature causes disease, and it is not a clinical explanation.

### 1.10 Locked holdout, evaluated once

| Model | ROC-AUC | PR-AUC | Recall | Precision | F1 | Acc. | TN | FP | FN | TP |
|-------|---------|--------|--------|-----------|-----|------|----|----|----|----|
| Logistic Regression | 0.8998 | 0.9021 | 0.9020 | 0.8364 | 0.8679 | 0.8478 | 64 | 18 | 10 | 92 |
| Random Forest | 0.8974 | 0.8937 | 0.9118 | 0.8532 | 0.8815 | 0.8641 | 66 | 16 | 9 | 93 |
| XGBoost | 0.8972 | 0.9040 | 0.8922 | 0.8349 | 0.8626 | 0.8424 | 64 | 18 | 11 | 91 |

All three land within 0.008 of their own 25-fold CV ROC-AUC, and within 0.003 of each other. That convergence is reassuring about the CV procedure, **not** a ranking — 184 rows cannot separate models that 25 folds also fail to separate.

---

## 2. Persisted artifacts

### 2.1 Model pipelines

| Artifact | Size | Estimator | Steps |
|----------|------|-----------|-------|
| `models/logistic_regression_pipeline.joblib` | 5.6 KB | `LogisticRegression` | `preprocess`, `model` |
| `models/random_forest_pipeline.joblib` | 11.9 MB | `RandomForestClassifier` | `preprocess`, `model` |
| `models/xgboost_pipeline.joblib` | 667.1 KB | `XGBClassifier` | `preprocess`, `model` |
| `models/tuned/logistic_regression_pipeline.joblib` | 5.6 KB | `LogisticRegression` | `preprocess`, `model` |
| `models/tuned/random_forest_pipeline.joblib` | 3.2 MB | `RandomForestClassifier` | `preprocess`, `model` |
| `models/tuned/xgboost_pipeline.joblib` | 355.7 KB | `XGBClassifier` | `preprocess`, `model` |

The Stage 3 `tuned/` artefacts are the ones intended for any future serving layer. Each carries its `ColumnTransformer` and estimator in a single object, so preprocessing cannot be skipped or reordered at inference time.

### 2.2 Metrics tables

| Artifact | Contents |
|----------|----------|
| `models/cv_metrics_overall.csv` | 25-fold CV, un-tuned baselines, mean ± SD |
| `models/cv_metrics_per_fold.csv` | Un-tuned, per-fold rows (enables paired comparison) |
| `models/cv_metrics_tuned_overall.csv` | 25-fold CV, selected configurations |
| `models/cv_metrics_tuned_per_fold.csv` | Selected, per-fold rows |
| `models/cv_metrics_by_cohort.csv` | Un-tuned per-cohort CV with reliability flags |
| `models/cv_metrics_tuned_by_cohort.csv` | Selected per-cohort CV with reliability flags |
| `models/threshold_analysis.csv` | Full 5-point threshold grid, 15 rows |
| `models/calibration_metrics.csv` | Brier, ECE, slope, intercept per model |
| `models/calibration_bins.csv` | 30 predicted-vs-observed bins |
| `models/feature_ablation_ca.csv` | `ca` ablation, both variants × 3 models |
| `models/feature_ablation_ca_by_cohort.csv` | Ablation split by cohort, 24 rows |
| `models/holdout_final_metrics.csv` | Single locked holdout evaluation |

### 2.3 Explainability (`models/explainability/`, 44 files)

* `explainability_summary.json` — attribution caveat, SHAP baseline definition, per-model output scale, additivity record, global tables, individual explanations, file inventory, console log.
* `{model}_shap_global.csv` — mean absolute SHAP per feature with direction.
* `{model}_feature_importance.csv` — permutation importance.
* `logistic_regression_coefficients.csv` — signed coefficients with odds ratios.
* `shap_summary_{model}.png`, `shap_global_bar_{model}.png` — global plots.
* `shap_individual_{model}_{case}.csv` / `.png` — 15 case explanations.
* `global_feature_importance.png` — cross-model importance comparison.

### 2.4 Reports and plots

* `models/stage3_report.json` — consolidated machine-readable record: data fingerprint, seed, CV config, holdout policy, both CV tables, OOF policy, search metadata and coverage, threshold grid and results, calibration, ablation, error analysis, explainability inventory, environment card, and the final holdout result.
* `models/calibration_curve.png` — reliability diagram, 97.8 KB.
* `models/baseline_report.json`, `baseline_metrics_overall.csv`, `baseline_metrics_by_cohort.csv`, `split_summary.csv` — Stage 2 records, retained.
* `notebooks/03_model_validation_explainability.ipynb` — 74 cells (40 markdown, 34 code), runs end to end.

---

## 3. Validation methodology

### 3.1 Data isolation

```
920 rows ──► locked 80/20 split (cohort × label stratified, seed 42)
              ├── 736 train ──► 5-fold × 5-repeat StratifiedKFold (25 folds)
              │                  ├── un-tuned baselines
              │                  ├── bounded search (first repeat only)
              │                  └── selected configurations
              │                        └── OOF predictions → threshold,
              │                            calibration, error analysis, cases
              └── 184 holdout ──► touched ONCE, after every decision was frozen
```

Audit evidence for isolation:

* CV fold indices are confined to `range(736)`. No fold index reaches the holdout.
* Every training row appears in validation at least once; no training row is dropped.
* The CV frame has 736 rows; index overlap with the holdout is `0`.
* The only function in `src/validation.py` accepting `X_test` / `y_test` is `final_holdout_evaluation()`, and the driver calls it after all five development artifacts are written.
* Calibration tables record `n = 736`; ablation tables record `n_folds = 25` — neither was computed on holdout rows.

### 3.2 Leakage prevention

Everything that learns from data — medians, scaler parameters, one-hot categories — lives **inside** the `Pipeline`. Each fold receives a fresh `clone`, so those statistics are re-fitted on that fold's training rows only.

Audit evidence for the fitted artifact:

* The persisted imputer medians equal the **train-slice** medians.
* They differ from the **full-dataset** medians, so a leak would be visible rather than silent.
* 13 raw features expand to 33 transformed columns; no NaN survives; the row count is never reduced.

### 3.3 Single-patient inference check

A synthetic patient absent from the file, deliberately carrying an invalid sentinel (`chol == 0`), a missing `ca` and a missing categorical, was passed through the persisted Logistic Regression pipeline:

* `(1, 13)` input → `(1, 33)` transformed, 0 NaN → `(1, 2)` probabilities, all finite.
* Prediction is identical scored alone and inside a 26-row batch → **order-invariant**.
* `predict()` left every fitted statistic byte-identical → **stateless at inference**.
* Invalid sentinels were masked and imputed by the same fitted transformers used at training time.

### 3.4 Out-of-fold policy

Each row's out-of-fold score is the **mean of its five per-repeat predictions**; in every repeat the model scoring that row had never seen it. This policy is recorded in `models/stage3_report.json` → `oof_prediction_policy` and asserted by the verification suite.

### 3.5 Paired comparison

Per-fold metrics are persisted for both baseline and selected configurations so model-vs-model comparison can be **paired** on identical folds. A pair is reported as separated only when the mean gap is at least as large as its own paired SD *and* the same side wins ≥80% of folds.

### 3.6 Verification suite

`python -m src.verify_stage3` runs **69 assertions** across six groups:

| Group | Coverage |
|-------|----------|
| A. Reproducibility | Raw CSV checksum, single seed constant, Stage 3 split reproduces Stage 2 exactly |
| B. Leakage safety | Train/test disjointness, CV folds confined to train, full validation coverage, preprocessing inside the pipeline, imputer medians are fold-train medians, identical OOF reproduction on identical folds (max diff `0.00e+00`) |
| C. Preprocessing correctness | No dropped rows, no surviving NaN, sentinel masking, forbidden columns absent, `ca` ablation removes the feature from all three lists |
| D. Artefacts and metric hygiene | Tuned pipelines reload and predict, artefact completeness, SHAP additivity, output scales declared, case coverage, OOF policy, threshold grid, confusion counts, Brier, calibration slope/intercept, ablation variants, reliability flag, holdout policy |
| E. Dependency pinning | No unpinned entries, all six key libraries pinned |

---

## 4. Known limitations

1. **920 rows, four cohorts.** Every metric carries wide uncertainty. After the paired test, two of three model pairs are no longer separated on ROC-AUC at all, so the models are not ranked as better or worse.

2. **One holdout, evaluated once.** 25-fold CV provides development estimates; the 184-row holdout is a single look and is now spent. Its numbers carry the uncertainty of one 184-row sample.

3. **Cohorts are not clinically comparable.** Prevalence spans 36.2% to 93.9%, so pooled metrics average over populations that differ in referral pathway and measurement. Switzerland is flagged unreliable at every model (92 positives, 6 negatives).

4. **`ca` is an angiographic measurement.** It correlates 0.516 with `num` — genuine information rather than a copied label, but part of the apparent strength comes from a measurement a screening setting would not have. Removing it costs 0.005–0.008 ROC-AUC.

5. **Search was partial for two models.** Random Forest scored 20 of 288 grid combinations and XGBoost 20 of 1296, sampled with replacement on the first CV repeat only. A different draw could select different parameters.

6. **Tuning deltas are biased upward.** The before/after comparison uses the same data that chose the hyper-parameters, so it cannot be read as evidence that tuning helped.

7. **Random Forest probabilities are under-dispersed.** Calibration slope 1.33, ECE 0.051 — roughly twice the others. It is deliberately **not** recalibrated, because recalibrating after seeing these results would be a modelling decision made on the evaluation data. Any future serving layer that exposes RF scores should surface this.

8. **Threshold 0.50 is a default, not a clinical choice.** No clinical threshold was specified. The grid shows ~9 sensitivity points are available for ~17 specificity points, but that trade-off is a clinical decision that has not been made.

9. **SHAP magnitudes are not comparable across model families.** Log-odds for Logistic Regression and XGBoost, probability for Random Forest.

10. **`chol == 0` recoding is an interpretation.** Reading those 172 rows as "not recorded" is well supported (they cluster entirely in two cohorts) but it is a judgement call, documented rather than silent.

11. **No external validation.** Nothing has been tested outside this file. Transport to a new population, measurement protocol or referral pathway is unestablished.

12. **Research prototype for risk assessment.** Not a medical device, not a diagnostic system. These numbers must not be presented as clinical capability.

---

## 5. Reproducibility status

| Property | Status | Evidence |
|----------|--------|----------|
| Random seed | Fixed | `RANDOM_SEED = 42`, single constant in `src/preprocessing.py` |
| Dependencies | Fully pinned | 9 entries in `requirements.txt`, no unpinned packages |
| Raw data integrity | Verified | SHA-256 `574f2fa2b43012fa25fca4fcdb36bd7c6bccdd0af4242f6c0e4c8633c5a072ae` matches on every load; audit re-checked independently |
| Raw data mutation | None | 79,346 bytes; `src/preprocessing.py` raises if the checksum changes |
| Split stability | Verified | Stage 2 and Stage 3 split entry points return identical indices |
| CV determinism | Verified | Identical folds reproduce identical OOF predictions, max diff `0.00e+00` |
| Model refit | Verified | Refitting produces bit-identical coefficients |
| SHAP determinism | Verified | Persisted vs recomputed global SHAP table: max diff `9.8e-17` across 33 features × 3 models |
| SHAP additivity | Verified | `1.8e-15` / `1.4e-15` / `4.7e-06` against each model's own output |
| Inference statelessness | Verified | `predict()` mutates no fitted statistic; batch-invariant |
| Artefact completeness | Verified | 69/69 assertions, including file-existence checks for every artifact |
| Environment recorded | Verified | Python 3.13.2, scikit-learn 1.7.1, numpy 2.3.1, pandas 2.3.2, shap 0.52.0, Windows 11 |
| Audit itself | Read-only | No file under `models/`, `data/`, `src/`, `notebooks/` modified |

**Environment used for the accepted run** (recorded in `models/stage3_report.json`):

```
python 3.13.2 | scikit-learn 1.7.1 | numpy 2.3.1 | pandas 2.3.2 | shap 0.52.0
platform: Windows-11-10.0.26200-SP0
generated_utc: 2026-10-01T06:38:31+00:00
```

Reproduction on a different OS or dependency version may produce numerically different artifacts. The methodology and every recorded decision will be identical; the byte-level outputs are seed-dependent and environment-sensitive.

---

## 6. Exact command to reproduce Stage 3

From the project root (`C:\Users\Asus\OneDrive\Documents\SW 2.0`):

```bash
pip install -r requirements.txt
python -m src.validate_stage3
```

This single command reproduces the entire Stage 3 pipeline in order:

1. Rebuild the Stage 2 locked split (736 / 184).
2. 25-fold cross-validation of un-tuned baselines.
3. Bounded hyper-parameter search per model.
4. 25-fold cross-validation of selected configurations.
5. Threshold analysis, calibration, `ca` ablation, error analysis.
6. Explainability artifacts, with additivity verified.
7. Persist selected pipelines to `models/tuned/`.
8. **Only then** evaluate the frozen pipelines on the 184-row holdout.
9. Invoke the verification suite (`69` assertions).

Verification can be run independently at any time:

```bash
python -m src.verify_stage3
```

To reproduce the Stage 2 baselines the Stage 3 split is anchored to:

```bash
python -m src.train
```

### Expected output

```
69/69 checks passed
```

Runtime on the recorded environment is approximately 4–6 minutes, dominated by the Random Forest search and SHAP computation.

### Files written by `python -m src.validate_stage3`

```
models/cv_metrics_overall.csv
models/cv_metrics_per_fold.csv
models/cv_metrics_by_cohort.csv
models/cv_metrics_tuned_overall.csv
models/cv_metrics_tuned_per_fold.csv
models/cv_metrics_tuned_by_cohort.csv
models/threshold_analysis.csv
models/calibration_metrics.csv
models/calibration_bins.csv
models/calibration_curve.png
models/feature_ablation_ca.csv
models/feature_ablation_ca_by_cohort.csv
models/holdout_final_metrics.csv
models/stage3_report.json
models/tuned/logistic_regression_pipeline.joblib
models/tuned/random_forest_pipeline.joblib
models/tuned/xgboost_pipeline.joblib
models/explainability/            (44 files)
```

---

## 7. Audit scope statement

* **No methodology was changed.** No model, feature, hyper-parameter, threshold, fold or preprocessing rule was modified during or after the audit.
* **No existing result was overwritten.** The audit read persisted artifacts and recomputed values in memory for comparison only. Every file under `models/` retains its timestamp from the accepted Stage 3 run.
* **No deployment work was performed.** `app/` is empty. No API, no dashboard, no service.
* **Stages 1–3 are frozen.** Any change to Stage 3 methodology requires explicit authorisation and a fresh audit.
