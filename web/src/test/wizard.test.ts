import { describe, expect, it } from 'vitest'
import {
  missingRequired,
  toRequest,
  toReviewRows,
  type FormValues,
} from '../state/wizard'
import type { PatientRiskRequest } from '../api/types'

const COMPLETE: FormValues = {
  age: '52',
  sex: 'Male',
  cp: 'typical angina',
  trestbps: '140',
  chol: '250',
  thalach: '100',
  oldpeak: '1.5',
  ca: '2',
  restecg: 'normal',
  slope: 'upsloping',
  thal: 'reversable defect',
  fbs: true,
  exang: false,
}

describe('toRequest', () => {
  it('maps the recognised values onto the API field names', () => {
    const body = toRequest(COMPLETE) as PatientRiskRequest
    expect(body.age).toBe(52)
    expect(body.sex).toBe('Male')
    expect(body.cp).toBe('typical angina')
    expect(body.trestbps).toBe(140)
    expect(body.chol).toBe(250)
    expect(body.thalach).toBe(100)
    expect(body.oldpeak).toBe(1.5)
    expect(body.ca).toBe(2)
    expect(body.restecg).toBe('normal')
    expect(body.slope).toBe('upsloping')
    expect(body.thal).toBe('reversable defect')
    expect(body.fbs).toBe(true)
    expect(body.exang).toBe(false)
  })

  it('sends thalach under the canonical key, with no thalch key present', () => {
    const body = toRequest(COMPLETE) as unknown as Record<string, unknown>
    expect(body).toHaveProperty('thalach')
    expect(body).not.toHaveProperty('thalch')
  })

  it('sends null for a field marked unavailable, never a substituted value', () => {
    const body = toRequest({ ...COMPLETE, chol: null, thal: null, fbs: null })
    expect(body.chol).toBeNull()
    expect(body.thal).toBeNull()
    expect(body.fbs).toBeNull()
  })

  it('never turns a missing numeric field into zero', () => {
    const body = toRequest({ ...COMPLETE, trestbps: null, oldpeak: '' })
    expect(body.trestbps).toBeNull()
    expect(body.oldpeak).toBeNull()
    expect(body.trestbps).not.toBe(0)
  })

  it('preserves a negative oldpeak, which the API accepts', () => {
    const body = toRequest({ ...COMPLETE, oldpeak: '-2.5' })
    expect(body.oldpeak).toBe(-2.5)
  })

  it('passes a genuine zero through rather than treating it as missing', () => {
    const body = toRequest({ ...COMPLETE, oldpeak: '0' })
    expect(body.oldpeak).toBe(0)
    expect(body.oldpeak).not.toBeNull()
  })

  it('rejects a non-numeric value by sending null instead of NaN', () => {
    const body = toRequest({ ...COMPLETE, trestbps: 'abc' })
    expect(body.trestbps).toBeNull()
  })

  it('omits nothing: all 13 keys are present in the payload', () => {
    const body = toRequest(COMPLETE) as unknown as Record<string, unknown>
    expect(Object.keys(body)).toHaveLength(13)
  })
})

describe('missingRequired', () => {
  it('reports all three required fields when empty', () => {
    expect(missingRequired({}).sort()).toEqual(['age', 'cp', 'sex'])
  })

  it('reports nothing once age, sex and cp are filled', () => {
    expect(missingRequired({ age: '52', sex: 'Male', cp: 'asymptomatic' })).toEqual(
      [],
    )
  })
})

describe('toReviewRows', () => {
  it('labels an absent optional field as Not available', () => {
    const rows = toReviewRows({ age: '52', sex: 'Male', cp: 'non-anginal' })
    const chol = rows.find((r) => r.field.apiKey === 'chol')
    expect(chol?.available).toBe(false)
    expect(chol?.display).toBe('Not available')
  })

  it('renders booleans as words rather than raw true/false', () => {
    const rows = toReviewRows(COMPLETE)
    expect(rows.find((r) => r.field.apiKey === 'fbs')?.display).toBe('Yes')
    expect(rows.find((r) => r.field.apiKey === 'exang')?.display).toBe('No')
  })

  it('renders a unit beside a measurement', () => {
    const rows = toReviewRows(COMPLETE)
    expect(
      rows.find((r) => r.field.apiKey === 'chol')?.display,
    ).toBe('250 mg/dL')
    expect(
      rows.find((r) => r.field.apiKey === 'thalach')?.display,
    ).toBe('100 bpm')
    expect(
      rows.find((r) => r.field.apiKey === 'trestbps')?.display,
    ).toBe('140 mmHg')
  })

  it('labels review rows with the plain-English label', () => {
    const rows = toReviewRows(COMPLETE)
    const chol = rows.find((r) => r.field.apiKey === 'chol')
    expect(chol?.field.label).toBe('Cholesterol level')
    expect(chol?.field.clinicalTerm).toBe('Serum cholesterol')
  })

  it('shows the human label while keeping the wire value separate', () => {
    const rows = toReviewRows(COMPLETE)
    const thal = rows.find((r) => r.field.apiKey === 'thal')
    expect(thal?.display).toBe('Reversible defect')
    expect(thal?.value).toBe('reversable defect')
  })
})
