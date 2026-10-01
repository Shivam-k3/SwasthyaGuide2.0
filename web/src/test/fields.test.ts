import { describe, expect, it } from 'vitest'
import { FIELDS, REQUIRED_API_KEYS, fieldDef, fieldsInGroup } from '../config/fields'

/**
 * These tests pin the frontend registry to the FastAPI contract. If the backend
 * enum values ever change, this fails rather than silently sending a value the
 * API would reject with a 422.
 */

describe('field registry', () => {
  it('exposes exactly the 13 backend fields', () => {
    expect(FIELDS).toHaveLength(13)
    expect(new Set(FIELDS.map((f) => f.apiKey)).size).toBe(13)
  })

  it('uses the canonical API key thalach, never thalch', () => {
    expect(fieldDef('thalach')).toBeDefined()
    expect(fieldDef('thalch')).toBeUndefined()
    expect(FIELDS.map((f) => f.apiKey)).not.toContain('thalch')
  })

  it('marks exactly age, sex and cp as required', () => {
    const required = FIELDS.filter((f) => f.required).map((f) => f.apiKey)
    expect(required.sort()).toEqual([...REQUIRED_API_KEYS].sort())
  })

  it('treats every non-required field as nullable, and no required field as such', () => {
    for (const field of FIELDS) {
      if (field.required) {
        expect(field.nullable, `${field.apiKey} should not be nullable`).toBe(false)
      } else {
        expect(field.nullable, `${field.apiKey} should be nullable`).toBe(true)
      }
    }
  })

  it('carries the backend categorical values verbatim', () => {
    const valuesOf = (key: string) =>
      fieldDef(key)?.options?.map((o) => o.value) ?? []

    expect(valuesOf('sex').sort()).toEqual(['Female', 'Male'])
    expect(valuesOf('cp').sort()).toEqual([
      'asymptomatic',
      'atypical angina',
      'non-anginal',
      'typical angina',
    ])
    expect(valuesOf('restecg').sort()).toEqual([
      'lv hypertrophy',
      'normal',
      'st-t abnormality',
    ])
    expect(valuesOf('slope').sort()).toEqual([
      'downsloping',
      'flat',
      'upsloping',
    ])
    // The dataset spells this "reversable". The UI label is corrected for
    // reading, but the wire value must stay as the backend defines it.
    expect(valuesOf('thal').sort()).toEqual([
      'fixed defect',
      'normal',
      'reversable defect',
    ])
  })

  it('does not give oldpeak a minimum, because negative ST depression is valid', () => {
    expect(fieldDef('oldpeak')?.min).toBeUndefined()
  })

  it('constrains ca to the 0-3 range the API enforces', () => {
    expect(fieldDef('ca')?.min).toBe(0)
    expect(fieldDef('ca')?.max).toBe(3)
    expect(fieldDef('ca')?.step).toBe(1)
  })

  it('places every field in a group that exists', () => {
    for (const group of [
      'basic',
      'chestPain',
      'measurements',
      'ecg',
      'additional',
    ] as const) {
      expect(fieldsInGroup(group).length).toBeGreaterThan(0)
    }
  })
})

describe('two-layer language', () => {
  it('gives every field a plain-English label and description', () => {
    for (const field of FIELDS) {
      expect(field.label.trim(), `${field.apiKey} label`).not.toBe('')
      expect(field.description.trim(), `${field.apiKey} description`).not.toBe('')
    }
  })

  it('keeps a clinical term for every field that has one', () => {
    const withTerm = FIELDS.filter((f) => f.clinicalTerm)
    // Age and sex need no clinical term; the other 11 carry one.
    expect(withTerm.map((f) => f.apiKey).sort()).toEqual([
      'ca',
      'chol',
      'cp',
      'exang',
      'fbs',
      'oldpeak',
      'restecg',
      'slope',
      'thal',
      'thalach',
      'trestbps',
    ])
    for (const field of withTerm) {
      expect(field.clinicalTerm?.trim(), field.apiKey).not.toBe('')
    }
    expect(fieldDef('age')?.clinicalTerm).toBeUndefined()
    expect(fieldDef('sex')?.clinicalTerm).toBeUndefined()
  })

  it('states the exact clinical definitions from the model', () => {
    expect(fieldDef('oldpeak')?.clinicalTerm).toBe('ST depression')
    expect(fieldDef('exang')?.clinicalTerm).toBe('Exercise-induced angina')
    expect(fieldDef('thalach')?.clinicalTerm).toBe('Maximum heart rate')
    expect(fieldDef('chol')?.clinicalTerm).toBe('Serum cholesterol')
    expect(fieldDef('trestbps')?.clinicalTerm).toBe('Resting blood pressure')
    expect(fieldDef('slope')?.clinicalTerm).toBe('Peak exercise ST slope')
    expect(fieldDef('ca')?.clinicalTerm).toBe('Number of major vessels (CA)')
    expect(fieldDef('cp')?.clinicalTerm).toBe('Chest pain type')
    // Fasting blood sugar uses a definition line rather than a bare term.
    expect(fieldDef('fbs')?.clinicalTerm).toBe('above 120 mg/dL')
    expect(fieldDef('fbs')?.clinicalTermPrefix).toBe(
      'Clinical definition used by this model',
    )
  })

  it('uses the plain-English labels required by the brief', () => {
    expect(fieldDef('age')?.label).toBe('Age')
    expect(fieldDef('sex')?.label).toBe('Sex')
    expect(fieldDef('cp')?.label).toBe('Type of chest discomfort')
    expect(fieldDef('exang')?.label).toBe('Chest pain during exercise')
    expect(fieldDef('trestbps')?.label).toBe('Blood pressure at rest')
    expect(fieldDef('chol')?.label).toBe('Cholesterol level')
    expect(fieldDef('thalach')?.label).toBe('Highest heart rate during exercise')
    expect(fieldDef('oldpeak')?.label).toBe('ECG change during exercise')
    expect(fieldDef('restecg')?.label).toBe('Resting heart test (ECG)')
    expect(fieldDef('slope')?.label).toBe('ECG pattern during exercise')
    expect(fieldDef('fbs')?.label).toBe('Fasting blood sugar')
    expect(fieldDef('ca')?.label).toBe('Major heart blood vessels with narrowing')
    expect(fieldDef('thal')?.label).toBe('Blood test result (thalassemia)')
  })

  it('warns that angiogram and thalassemia are usually unavailable', () => {
    expect(fieldDef('ca')?.unavailableNote).toBe(
      'Usually unavailable unless this test was performed.',
    )
    expect(fieldDef('thal')?.unavailableNote).toBe(
      'Usually unavailable unless this test was performed.',
    )
  })

  it('does not push clinical vocabulary into the primary label', () => {
    // The plain layer must stay readable; jargon belongs to the term line.
    for (const field of FIELDS) {
      if (field.clinicalTerm && field.clinicalTerm !== field.label) {
        expect(field.label.toLowerCase()).not.toBe(
          field.clinicalTerm!.toLowerCase(),
        )
      }
    }
  })

  it('still groups fields into the five required sections', () => {
    const keysOf = (g: string) =>
      fieldsInGroup(g as never).map((f) => f.apiKey)
    expect(keysOf('basic')).toEqual(['age', 'sex'])
    expect(keysOf('chestPain')).toEqual(['cp', 'exang'])
    expect(keysOf('measurements')).toEqual(['trestbps', 'chol', 'thalach', 'fbs'])
    expect(keysOf('ecg')).toEqual(['restecg', 'oldpeak', 'slope'])
    expect(keysOf('additional')).toEqual(['ca', 'thal'])
  })
})
