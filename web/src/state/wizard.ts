import type { PatientRiskRequest } from '../api/types'
import { FIELDS, fieldDef, labelForOption, type FieldDef } from '../config/fields'

/**
 * Values held by the form while a person is filling it in.
 *
 * `null` means "not available", which is an explicit choice the person makes and
 * which is sent to the API as `null`. It is never replaced by a zero, a mean,
 * or any other guess. Missing data is imputed by the model pipeline, with an
 * indicator that the value was absent, and the API says so in its notes.
 */
export type FormValues = Record<string, string | boolean | null>

export type Step = 'form' | 'review' | 'result'

export interface WizardState {
  step: Step
  values: FormValues
}

export type WizardAction =
  | { type: 'edit' }
  | { type: 'review' }
  | { type: 'result' }
  | { type: 'restart' }
  | { type: 'setValues'; values: FormValues }

export const initialWizardState: WizardState = { step: 'form', values: {} }

export function wizardReducer(
  state: WizardState,
  action: WizardAction,
): WizardState {
  switch (action.type) {
    case 'edit':
      return { ...state, step: 'form' }
    case 'review':
      return { ...state, step: 'review' }
    case 'result':
      return { ...state, step: 'result' }
    case 'setValues':
      return { ...state, values: action.values }
    case 'restart':
      return initialWizardState
  }
}

export interface ReviewRow {
  field: FieldDef
  /** null means the person marked it not available. */
  value: string | boolean | null
  display: string
  available: boolean
}

/**
 * Build the review list. This only formats what was entered; it performs no
 * medical reasoning.
 */
export function toReviewRows(values: FormValues): ReviewRow[] {
  return FIELDS.map((field) => {
    const raw = values[field.apiKey] ?? null
    const available = raw !== null && raw !== undefined && raw !== ''
    let display = 'Not available'
    if (available) {
      if (typeof raw === 'boolean') {
        display = raw ? 'Yes' : 'No'
      } else if (field.control === 'enum') {
        display = labelForOption(field, String(raw)) ?? String(raw)
      } else {
        display = field.unit ? `${raw} ${field.unit}` : String(raw)
      }
    }
    return { field, value: available ? raw : null, display, available }
  })
}

/**
 * Serialise form values into the POST /predict body.
 *
 * Only the 13 backend fields are sent, using the API's own key names and its
 * exact enum strings. Nothing is imputed, defaulted, or invented here.
 */
export function toRequest(values: FormValues): PatientRiskRequest {
  const readString = (key: string): string | null => {
    const raw = values[key]
    if (raw === null || raw === undefined) return null
    const text = String(raw).trim()
    return text === '' ? null : text
  }
  const readNumber = (key: string): number | null => {
    const text = readString(key)
    if (text === null) return null
    const parsed = Number(text)
    return Number.isFinite(parsed) ? parsed : null
  }
  const readBoolean = (key: string): boolean | null => {
    const raw = values[key]
    return typeof raw === 'boolean' ? raw : null
  }

  const body: Record<string, unknown> = {}

  for (const field of FIELDS) {
    const key = field.apiKey
    if (field.control === 'number') {
      const value = readNumber(key)
      // Required numeric fields must always be present; the schema enforces it.
      body[key] = value
    } else if (field.control === 'boolean') {
      body[key] = readBoolean(key)
    } else {
      body[key] = readString(key)
    }
  }

  return body as unknown as PatientRiskRequest
}

/** Required fields the form could not fill, for the pre-submit check. */
export function missingRequired(values: FormValues): string[] {
  return ['age', 'sex', 'cp'].filter((key) => {
    const field = fieldDef(key)
    if (!field) return false
    const raw = values[key]
    return raw === null || raw === undefined || raw === ''
  })
}
