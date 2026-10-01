/**
 * TypeScript mirrors of the FastAPI schemas in app/schemas.py.
 *
 * These are hand-written from the generated OpenAPI document, not generated at
 * build time. Every value here is the backend's own contract: wire values are
 * never renamed, normalised, or re-interpreted by the frontend.
 */

export type SexValue = 'Female' | 'Male'
export type ChestPainValue =
  | 'asymptomatic'
  | 'atypical angina'
  | 'non-anginal'
  | 'typical angina'
export type RestEcgValue = 'lv hypertrophy' | 'normal' | 'st-t abnormality'
export type SlopeValue = 'downsloping' | 'flat' | 'upsloping'
export type ThalValue = 'fixed defect' | 'normal' | 'reversable defect'

export type PredictionLabel = 'elevated_risk' | 'lower_predicted_risk'

/** Request body for POST /predict. */
export interface PatientRiskRequest {
  age: number
  sex: SexValue
  cp: ChestPainValue
  trestbps?: number | null
  chol?: number | null
  thalach?: number | null
  oldpeak?: number | null
  ca?: number | null
  restecg?: RestEcgValue | null
  slope?: SlopeValue | null
  thal?: ThalValue | null
  fbs?: boolean | null
  exang?: boolean | null
}

export interface ExpectedOperatingPoint {
  source: string
  n_rows: number
  n_positive: number
  threshold: number
  recall: number
  specificity: number
  precision: number
  f1: number
  false_positive_rate: number
  false_negative_rate: number
  roc_auc: number
  pr_auc: number
  tn: number
  fp: number
  fn: number
  tp: number
}

export interface RiskPredictionResponse {
  model: string
  model_version: string
  risk_probability: number
  prediction: PredictionLabel
  threshold: number
  message: string
  notes: string[]
  expected_operating_point: ExpectedOperatingPoint
  disclaimer: string
}

export interface FeatureSpec {
  name: string
  kind: string
  required: boolean
  description: string
}

export interface ModelInfoResponse {
  model: string
  model_type: string
  /** Repo-relative artifact path. Never rendered in the UI. */
  artifact: string
  artifact_stage: string
  provenance: string
  deployment_threshold: number
  threshold_source: string
  threshold_policy: string
  n_input_features: number
  input_features: FeatureSpec[]
  n_transformed_features: number
  available_models: string[]
  selection_rationale: string
  calibration_note: string
  /** Backend runtime metadata. Never rendered in the UI. */
  environment: Record<string, unknown>
  disclaimer: string
}

export interface HealthResponse {
  status: 'ok' | 'degraded'
  api_stage: string
  api_version: string
  model_loaded: boolean
  model: string
  deployment_threshold: number
  disclaimer: string
}

/** One entry of FastAPI's 422 `detail` array. */
export interface ApiValidationError {
  type?: string
  loc?: (string | number)[]
  msg?: string
  input?: unknown
  ctx?: Record<string, unknown>
}
