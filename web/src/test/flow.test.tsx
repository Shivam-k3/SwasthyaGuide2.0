import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { App } from '../App'

const HEALTH = {
  status: 'ok',
  api_stage: '4 - FastAPI serving layer',
  api_version: '4.0.0',
  model_loaded: true,
  model: 'logistic_regression',
  deployment_threshold: 0.4,
  disclaimer: 'Research prototype for health risk assessment.',
}

const MODEL_INFO = {
  model: 'logistic_regression',
  model_type: 'LogisticRegression',
  artifact: 'models/tuned/logistic_regression_pipeline.joblib',
  artifact_stage: 'Stage 3 selected configuration',
  provenance: 'Fitted in Stage 3 on the 736-row training portion.',
  deployment_threshold: 0.4,
  threshold_source: "Measured for 'logistic_regression' at this exact value.",
  threshold_policy: 'Chosen to favour recall over specificity.',
  n_input_features: 13,
  input_features: [],
  n_transformed_features: 33,
  available_models: ['logistic_regression', 'random_forest', 'xgboost'],
  selection_rationale: 'Default serving choice on the development estimate.',
  calibration_note: 'Closest to ideal calibration of the three.',
  environment: { python: '3.13.2' },
  disclaimer: 'Research prototype for health risk assessment.',
}

const PREDICTION = {
  model: 'logistic_regression',
  model_version: 'stage3-2026-10-01T06:38:31+00:00/data-574f2fa2b430',
  risk_probability: 0.558846,
  prediction: 'elevated_risk',
  threshold: 0.4,
  message:
    'Elevated predicted risk. Clinical evaluation may be appropriate. This is a research risk score, not a diagnosis.',
  notes: ['chol=0 was treated as not recorded and imputed with a missingness indicator.'],
  expected_operating_point: {
    source: 'Stage 3 out-of-fold predictions on the 736-row development pool.',
    n_rows: 736,
    n_positive: 407,
    threshold: 0.4,
    recall: 0.8968058968058968,
    specificity: 0.7355623100303952,
    precision: 0.8075221238938053,
    f1: 0.8498253783469151,
    false_positive_rate: 0.2644376899696048,
    false_negative_rate: 0.10319410319410319,
    roc_auc: 0.9037288186224357,
    pr_auc: 0.9110587355657116,
    tn: 242,
    fp: 87,
    fn: 42,
    tp: 365,
  },
  disclaimer:
    'Research prototype for health risk assessment. Not a medical device, not a diagnostic system.',
}

interface Route {
  match: (url: string, init?: RequestInit) => boolean
  respond: () => Response
}

function mockApi(routes: Route[]): void {
  vi.stubGlobal(
    'fetch',
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      for (const route of routes) {
        if (route.match(url, init)) return route.respond()
      }
      throw new TypeError('no route for ' + url)
    }),
  )
}

const ok = (body: unknown) =>
  new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'Content-Type': 'application/json' },
  })

function defaultRoutes(onPredict?: () => Response): Route[] {
  return [
    { match: (u) => u.includes('/health'), respond: () => ok(HEALTH) },
    { match: (u) => u.includes('/model-info'), respond: () => ok(MODEL_INFO) },
    {
      match: (u, init) => u.includes('/predict') && init?.method === 'POST',
      respond: onPredict ?? (() => ok(PREDICTION)),
    },
  ]
}

function renderApp(initialPath = '/assessment') {
  return render(
    <MemoryRouter initialEntries={[initialPath]}>
      <App />
    </MemoryRouter>,
  )
}

async function fillRequiredBasics(user: ReturnType<typeof userEvent.setup>) {
  await user.type(screen.getByLabelText(/^Age/), '52')
  await user.selectOptions(screen.getByLabelText(/^Sex/), 'Male')
  await user.selectOptions(
    screen.getByLabelText(/^Type of chest discomfort/),
    'typical angina',
  )
}

describe('assessment flow', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('renders the form with the 13 fields and required markers', async () => {
    mockApi(defaultRoutes())
    renderApp()

    expect(
      await screen.findByRole('heading', {
        name: /Cardiovascular risk assessment/i,
      }),
    ).toBeInTheDocument()

    // Number and enum fields are labelled with <label htmlFor>. These are the
    // plain-English primary labels; the clinical terms render as secondary text.
    for (const label of [
      /^Age/,
      /^Sex/,
      /^Type of chest discomfort/,
      /^Blood pressure at rest/,
      /^Cholesterol level/,
      /^Highest heart rate during exercise/,
      /^ECG change during exercise/,
      /^Resting heart test \(ECG\)/,
      /^ECG pattern during exercise/,
      /^Major heart blood vessels with narrowing/,
      /^Blood test result \(thalassemia\)/,
    ]) {
      expect(screen.getByLabelText(label)).toBeInTheDocument()
    }

    // Boolean fields are radio groups named by their legend, because a legend
    // cannot be a <label htmlFor>.
    for (const name of [/Chest pain during exercise/, /Fasting blood sugar/]) {
      expect(screen.getByRole('group', { name })).toBeInTheDocument()
    }
  })

  it('shows both a plain-English label and the clinical term for every field', async () => {
    mockApi(defaultRoutes())
    renderApp()
    await screen.findByRole('heading', { name: /Cardiovascular risk assessment/i })

    const text = document.body.textContent ?? ''
    // Primary plain-English layer.
    for (const plain of [
      'Type of chest discomfort',
      'Blood pressure at rest',
      'Cholesterol level',
      'Highest heart rate during exercise',
      'ECG change during exercise',
      'Resting heart test (ECG)',
      'ECG pattern during exercise',
      'Major heart blood vessels with narrowing',
      'Blood test result (thalassemia)',
    ]) {
      expect(text).toContain(plain)
    }
    // Secondary clinical layer, still visible for research/professional users.
    for (const clinical of [
      'Chest pain type',
      'Resting blood pressure',
      'Serum cholesterol',
      'Maximum heart rate',
      'ST depression',
      'Resting ECG',
      'Peak exercise ST slope',
      'Number of major vessels (CA)',
      'Thalassemia',
    ]) {
      expect(text).toContain(clinical)
    }
    // The clinical line is explicitly prefixed so the two layers stay distinct.
    expect(text).toContain('Clinical term:')
    expect(text).toContain('Clinical definition used by this model')
  })

  it('tells a person when a field is usually unavailable', async () => {
    mockApi(defaultRoutes())
    renderApp()
    await screen.findByRole('heading', { name: /Cardiovascular risk assessment/i })
    const text = document.body.textContent ?? ''
    expect(text).toContain('Usually unavailable unless this test was performed.')
    // "Not available" must remain offered.
    expect(screen.getAllByLabelText('Not available').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Not available').length).toBeGreaterThan(0)
  })

  it('blocks submission and reports missing required fields', async () => {
    mockApi(defaultRoutes())
    const user = userEvent.setup()
    renderApp()

    await user.click(
      await screen.findByRole('button', { name: /Continue to review/i }),
    )

    const alerts = await screen.findAllByRole('alert')
    expect(alerts.length).toBeGreaterThan(0)
    expect(screen.queryByText(/Run Assessment/i)).not.toBeInTheDocument()
  })

  it('sends null for fields left unavailable and never zero', async () => {
    // Capture the request body via a dedicated spy.
    const calls: { url: string; body?: string }[] = []
    vi.stubGlobal(
      'fetch',
      vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input)
        calls.push({ url, body: typeof init?.body === 'string' ? init.body : undefined })
        if (url.includes('/health')) return ok(HEALTH)
        if (url.includes('/model-info')) return ok(MODEL_INFO)
        if (url.includes('/predict')) return ok(PREDICTION)
        throw new TypeError('no route')
      }),
    )

    const user = userEvent.setup()
    renderApp()

    await fillRequiredBasics(user)
    await user.click(screen.getByRole('button', { name: /Continue to review/i }))
    await user.click(await screen.findByRole('button', { name: /Run Assessment/i }))

    await waitFor(() => {
      expect(calls.some((c) => c.url.includes('/predict'))).toBe(true)
    })

    const predictCall = calls.find((c) => c.url.includes('/predict'))
    expect(predictCall?.body).toBeTruthy()
    const payload = JSON.parse(predictCall!.body!) as Record<string, unknown>

    expect(payload.age).toBe(52)
    expect(payload.sex).toBe('Male')
    expect(payload.cp).toBe('typical angina')
    expect(payload.thalach).toBeNull()
    expect(payload).not.toHaveProperty('thalch')
    expect(payload.chol).toBeNull()
    expect(payload.trestbps).toBeNull()
    // Any optional field left blank must be null, never 0.
    for (const key of ['chol', 'trestbps', 'oldpeak', 'ca', 'fbs', 'exang']) {
      expect(payload[key]).toBeNull()
    }
  })

  it('shows the review step before anything is sent', async () => {
    mockApi(defaultRoutes())
    const user = userEvent.setup()
    renderApp()

    await fillRequiredBasics(user)
    await user.click(screen.getByRole('button', { name: /Continue to review/i }))

    expect(
      (await screen.findAllByText(/Review your entries/i)).length,
    ).toBeGreaterThan(0)
    expect(screen.getByText(/Nothing has been sent yet/i)).toBeInTheDocument()
    expect(screen.getAllByText(/Not available/i).length).toBeGreaterThan(0)
  })

  it('renders the backend result verbatim and never claims a diagnosis', async () => {
    mockApi(defaultRoutes())
    const user = userEvent.setup()
    renderApp()

    await fillRequiredBasics(user)
    await user.click(screen.getByRole('button', { name: /Continue to review/i }))
    await user.click(await screen.findByRole('button', { name: /Run Assessment/i }))

    expect(
      (await screen.findAllByText(/Elevated predicted risk/i)).length,
    ).toBeGreaterThan(0)
    // The probability is displayed as returned, formatted only.
    expect(screen.getByText('55.9%')).toBeInTheDocument()
    // Backend message verbatim.
    expect(screen.getByText(PREDICTION.message)).toBeInTheDocument()
    // Backend note rendered.
    expect(screen.getByText(PREDICTION.notes[0])).toBeInTheDocument()

    // Forbidden phrasings.
    expect(screen.queryByText(/you have heart disease/i)).not.toBeInTheDocument()
    expect(
      screen.queryByText(/you don't have heart disease/i),
    ).not.toBeInTheDocument()
    expect(screen.queryByText(/caused your/i)).not.toBeInTheDocument()
  })

  it('keeps Model Information collapsed until asked for', async () => {
    mockApi(defaultRoutes())
    const user = userEvent.setup()
    renderApp()

    await fillRequiredBasics(user)
    await user.click(screen.getByRole('button', { name: /Continue to review/i }))
    await user.click(await screen.findByRole('button', { name: /Run Assessment/i }))

    const toggle = await screen.findByRole('button', {
      name: /Model Information/i,
    })
    expect(toggle).toHaveAttribute('aria-expanded', 'false')

    await user.click(toggle)
    expect(toggle).toHaveAttribute('aria-expanded', 'true')
  })

  it('never exposes the artifact path or environment block', async () => {
    mockApi(defaultRoutes())
    const user = userEvent.setup()
    renderApp()

    await fillRequiredBasics(user)
    await user.click(screen.getByRole('button', { name: /Continue to review/i }))
    await user.click(await screen.findByRole('button', { name: /Run Assessment/i }))
    await screen.findAllByText(/Elevated predicted risk/i)

    const body = document.body.textContent ?? ''
    expect(body).not.toContain('models/tuned')
    expect(body).not.toContain('logistic_regression_pipeline.joblib')
    expect(body).not.toContain('3.13.2')
  })

  it('shows a clear degraded message on 503 without a stack trace', async () => {
    mockApi(
      defaultRoutes(
        () =>
          new Response(
            JSON.stringify({
              detail:
                'The prediction model is currently unavailable. Please retry later.',
            }),
            { status: 503, headers: { 'Content-Type': 'application/json' } },
          ),
      ),
    )
    const user = userEvent.setup()
    renderApp()

    await fillRequiredBasics(user)
    await user.click(screen.getByRole('button', { name: /Continue to review/i }))
    await user.click(await screen.findByRole('button', { name: /Run Assessment/i }))

    const alert = await screen.findByRole('alert')
    expect(
      within(alert).getByText(/currently unavailable/i),
    ).toBeInTheDocument()
    expect(document.body.textContent).not.toContain('Traceback')
  })

  it('maps a 422 back onto the form', async () => {
    mockApi(
      defaultRoutes(
        () =>
          new Response(
            JSON.stringify({
              detail: [
                {
                  type: 'literal_error',
                  loc: ['body', 'cp'],
                  msg: 'Input should be a valid chest pain category',
                },
              ],
            }),
            { status: 422, headers: { 'Content-Type': 'application/json' } },
          ),
      ),
    )
    const user = userEvent.setup()
    renderApp()

    await fillRequiredBasics(user)
    await user.click(screen.getByRole('button', { name: /Continue to review/i }))
    await user.click(await screen.findByRole('button', { name: /Run Assessment/i }))

    // The flow returns to the form so the flagged field can be corrected.
    await waitFor(() => {
      expect(
        screen.getByRole('button', { name: /Continue to review/i }),
      ).toBeInTheDocument()
    })
    // Singular when exactly one field is flagged, plural otherwise.
    expect(screen.getByText(/needs? attention/i)).toBeInTheDocument()
  })
})

describe('routing', () => {
  beforeEach(() => {
    vi.restoreAllMocks()
  })

  it('redirects an unknown route to the landing page', async () => {
    mockApi(defaultRoutes())
    renderApp('/does-not-exist')

    expect(
      await screen.findByRole('heading', {
        name: /Understand your cardiovascular risk/i,
      }),
    ).toBeInTheDocument()
  })

  it('renders the heart image from a same-origin asset path', async () => {
    mockApi(defaultRoutes())
    renderApp('/')

    const image = await screen.findByAltText(/human heart/i)
    expect(image).toHaveAttribute('src', '/assets/heart-hero.png')
  })

  it('does not reference the original filesystem path anywhere', async () => {
    mockApi(defaultRoutes())
    renderApp('/')
    await screen.findByAltText(/human heart/i)
    expect(document.body.innerHTML).not.toContain('C:\\Users')
    expect(document.body.innerHTML).not.toContain('Downloads')
  })
})
