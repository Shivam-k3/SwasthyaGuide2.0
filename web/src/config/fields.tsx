import type {
  ChestPainValue,
  RestEcgValue,
  SexValue,
  SlopeValue,
  ThalValue,
} from '../api/types'

/** How a field is rendered. */
export type ControlKind = 'number' | 'enum' | 'boolean'

export type FieldGroupId =
  | 'basic'
  | 'chestPain'
  | 'measurements'
  | 'ecg'
  | 'additional'

export interface FieldGroup {
  id: FieldGroupId
  title: string
  /** Short plain-English line explaining what the group collects. */
  blurb: string
}

export interface FieldOption<V extends string = string> {
  /** The exact value sent to FastAPI. Never altered for display. */
  value: V
  /** Human-readable label. Presentation only. */
  label: string
  hint?: string
}

/**
 * Two-layer field definition.
 *
 * `label` and `description` are written for a person with no medical training.
 * `clinicalTerm` keeps the exact clinical variable visible for research and
 * professional users. The `apiKey` is always the backend's own field name and
 * is never changed here.
 */
export interface FieldDef<V extends string = string> {
  /** API key. Names are the backend's own and are never altered. */
  apiKey: string
  /** Plain-English primary label. */
  label: string
  /** Plain-English explanation of what to enter. */
  description: string
  /** Clinical/medical term, kept as a secondary reference. */
  clinicalTerm?: string
  /** Prefix for the clinical line. Defaults to "Clinical term". */
  clinicalTermPrefix?: string
  /** Plain-English note for fields that are often not available. */
  unavailableNote?: string
  group: FieldGroupId
  unit?: string
  nullable: boolean
  control: ControlKind
  required?: boolean
  options?: readonly FieldOption<V>[]
  /** Presentational ordering within a group. */
  order: number
  min?: number
  max?: number
  /** Whole numbers only, e.g. age and ca. */
  step?: number
}

export const FIELD_GROUPS: readonly FieldGroup[] = [
  {
    id: 'basic',
    title: 'About you',
    blurb: 'The two details that frame every other reading.',
  },
  {
    id: 'chestPain',
    title: 'Chest symptoms',
    blurb: 'How and when the discomfort occurs is the strongest single signal in the model.',
  },
  {
    id: 'measurements',
    title: 'Basic measurements',
    blurb: 'Routine readings from a resting check-up or a blood test.',
  },
  {
    id: 'ecg',
    title: 'ECG and exercise test',
    blurb: 'Findings from a resting heart test and an exercise test.',
  },
  {
    id: 'additional',
    title: 'Additional clinical information',
    blurb: 'Results that are usually only available after a cardiac investigation. Mark any you do not have as not available.',
  },
]

export const FIELDS: readonly FieldDef[] = [
  {
    apiKey: 'age',
    label: 'Age',
    description: 'Enter your age in years.',
    group: 'basic',
    unit: 'years',
    nullable: false,
    control: 'number',
    required: true,
    order: 1,
    min: 1,
    max: 120,
    step: 1,
  },
  {
    apiKey: 'sex',
    label: 'Sex',
    description: 'Select the sex recorded for this assessment.',
    group: 'basic',
    nullable: false,
    control: 'enum',
    required: true,
    order: 2,
    options: [
      { value: 'Female', label: 'Female' },
      { value: 'Male', label: 'Male' },
    ] satisfies readonly FieldOption<SexValue>[],
  },
  {
    apiKey: 'cp',
    label: 'Type of chest discomfort',
    description:
      'Choose the option that best describes the chest discomfort recorded for this assessment.',
    clinicalTerm: 'Chest pain type',
    group: 'chestPain',
    nullable: false,
    control: 'enum',
    required: true,
    order: 1,
    options: [
      {
        value: 'asymptomatic',
        label: 'No chest discomfort',
        hint: 'No chest pain was reported.',
      },
      {
        value: 'atypical angina',
        label: 'Atypical angina',
        hint: 'Chest pain that does not fit the classic pattern.',
      },
      {
        value: 'non-anginal',
        label: 'Non-anginal pain',
        hint: 'Chest discomfort attributed to something other than the heart.',
      },
      {
        value: 'typical angina',
        label: 'Typical angina',
        hint: 'Chest pain with the classic pattern, relieved by rest.',
      },
    ] satisfies readonly FieldOption<ChestPainValue>[],
  },
  {
    apiKey: 'exang',
    label: 'Chest pain during exercise',
    description: 'Did you experience chest discomfort during the exercise test?',
    clinicalTerm: 'Exercise-induced angina',
    group: 'chestPain',
    nullable: true,
    control: 'boolean',
    order: 2,
  },
  {
    apiKey: 'trestbps',
    label: 'Blood pressure at rest',
    description: 'Enter the blood pressure measured while you were resting.',
    clinicalTerm: 'Resting blood pressure',
    group: 'measurements',
    unit: 'mmHg',
    nullable: true,
    control: 'number',
    order: 1,
    min: 0,
  },
  {
    apiKey: 'chol',
    label: 'Cholesterol level',
    description: 'Enter your total cholesterol level from your blood test.',
    clinicalTerm: 'Serum cholesterol',
    group: 'measurements',
    unit: 'mg/dL',
    nullable: true,
    control: 'number',
    order: 2,
    min: 0,
  },
  {
    apiKey: 'thalach',
    label: 'Highest heart rate during exercise',
    description:
      'Enter the highest heart rate recorded during your exercise test.',
    clinicalTerm: 'Maximum heart rate',
    group: 'measurements',
    unit: 'bpm',
    nullable: true,
    control: 'number',
    order: 3,
    min: 0,
  },
  {
    apiKey: 'fbs',
    label: 'Fasting blood sugar',
    description: 'Was your fasting blood sugar above 120 mg/dL?',
    clinicalTerm: 'above 120 mg/dL',
    clinicalTermPrefix: 'Clinical definition used by this model',
    group: 'measurements',
    nullable: true,
    control: 'boolean',
    order: 4,
  },
  {
    apiKey: 'restecg',
    label: 'Resting heart test (ECG)',
    description: 'Choose the result recorded from your resting ECG.',
    clinicalTerm: 'Resting ECG',
    group: 'ecg',
    nullable: true,
    control: 'enum',
    order: 1,
    options: [
      { value: 'normal', label: 'Normal' },
      { value: 'lv hypertrophy', label: 'Left ventricular hypertrophy' },
      { value: 'st-t abnormality', label: 'ST-T abnormality' },
    ] satisfies readonly FieldOption<RestEcgValue>[],
  },
  {
    apiKey: 'oldpeak',
    label: 'ECG change during exercise',
    description:
      'This is the change seen in the ST segment during an exercise ECG.',
    clinicalTerm: 'ST depression',
    group: 'ecg',
    nullable: true,
    control: 'number',
    order: 2,
    step: 0.1,
  },
  {
    apiKey: 'slope',
    label: 'ECG pattern during exercise',
    description: 'Choose the pattern recorded during your exercise test.',
    clinicalTerm: 'Peak exercise ST slope',
    group: 'ecg',
    nullable: true,
    control: 'enum',
    order: 3,
    options: [
      { value: 'upsloping', label: 'Upsloping' },
      { value: 'flat', label: 'Flat' },
      { value: 'downsloping', label: 'Downsloping' },
    ] satisfies readonly FieldOption<SlopeValue>[],
  },
  {
    apiKey: 'ca',
    label: 'Major heart blood vessels with narrowing',
    description:
      'If you have had an angiogram, enter the number of major heart blood vessels that were found to be narrowed.',
    unavailableNote: 'Usually unavailable unless this test was performed.',
    clinicalTerm: 'Number of major vessels (CA)',
    group: 'additional',
    unit: 'vessels',
    nullable: true,
    control: 'number',
    order: 1,
    min: 0,
    max: 3,
    step: 1,
  },
  {
    apiKey: 'thal',
    label: 'Blood test result (thalassemia)',
    description: 'If this blood test was performed, select the recorded result.',
    unavailableNote: 'Usually unavailable unless this test was performed.',
    clinicalTerm: 'Thalassemia',
    group: 'additional',
    nullable: true,
    control: 'enum',
    order: 2,
    options: [
      { value: 'normal', label: 'Normal' },
      { value: 'reversable defect', label: 'Reversible defect' },
      { value: 'fixed defect', label: 'Fixed defect' },
    ] satisfies readonly FieldOption<ThalValue>[],
  },
]

export const REQUIRED_API_KEYS = ['age', 'sex', 'cp'] as const

const BY_KEY = new Map(FIELDS.map((f) => [f.apiKey, f]))

export function fieldDef(apiKey: string): FieldDef | undefined {
  return BY_KEY.get(apiKey)
}

export function fieldsInGroup(group: FieldGroupId): FieldDef[] {
  return FIELDS.filter((f) => f.group === group).sort((a, b) => a.order - b.order)
}

export function labelForOption(
  field: FieldDef,
  value: string,
): string | undefined {
  return field.options?.find((o) => o.value === value)?.label
}
