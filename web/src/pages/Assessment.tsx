import { useEffect, useReducer, useState } from 'react'
import { ArrowLeft, ArrowRight, Info } from 'lucide-react'
import { ApiError, predict } from '../api/client'
import type { HealthResponse, ModelInfoResponse, RiskPredictionResponse } from '../api/types'
import { Button } from '../components/ui/Button'
import { ErrorNotice, PendingNote } from '../components/ui/ErrorNotice'
import { BooleanField } from '../components/assessment/BooleanField'
import { EnumField } from '../components/assessment/EnumField'
import { NumberField } from '../components/assessment/NumberField'
import { ReviewSummary } from '../components/assessment/ReviewSummary'
import { RiskResult } from '../components/result/RiskResult'
import {
  FIELDS,
  FIELD_GROUPS,
  REQUIRED_API_KEYS,
  fieldsInGroup,
  type FieldDef,
} from '../config/fields'
import {
  initialWizardState,
  missingRequired,
  toRequest,
  toReviewRows,
  wizardReducer,
  type FormValues,
  type Step,
} from '../state/wizard'

const STEP_LABELS = ['Enter details', 'Review', 'Result'] as const

/** Maps the wizard's step to its position in the progress sequence. */
const STEP_INDEX: Record<Step, number> = { form: 0, review: 1, result: 2 }

/**
 * Read one numeric entry as a number, or `null` when it is absent.
 *
 * An empty string, `null` and `undefined` all mean "not available" here; the
 * API receives `null` for those, never a coerced 0.
 */
function readNumber(values: FormValues, key: string): number | null {
  const raw = values[key]
  if (raw === null || raw === undefined || raw === '') return null
  const parsed = Number(String(raw))
  return Number.isNaN(parsed) ? null : parsed
}

/** Is a value absent rather than filled in? */
function isBlank(raw: string | boolean | null | undefined): boolean {
  return raw === null || raw === undefined || raw === ''
}

/**
 * Range check for one numeric field, mirroring the API's own constraints.
 *
 * Returns an error message, or `undefined` when the entry is acceptable. A field
 * marked unavailable is skipped: it is sent as `null` and never validated.
 */
function numberRangeError(field: FieldDef, parsed: number): string | undefined {
  if (field.min !== undefined && parsed < field.min) {
    return `Enter ${field.min} or more.`
  }
  if (field.max !== undefined && parsed > field.max) {
    return `Enter ${field.max} or less.`
  }
  if (field.step === 1 && !Number.isInteger(parsed)) {
    return 'Enter a whole number.'
  }
  return undefined
}

/**
 * The three stages share one route and one form state, so the result is never
 * reachable without passing through review, and nothing is submitted until the
 * person confirms.
 */
export function Assessment({
  modelInfo,
  health,
  healthError,
}: {
  modelInfo: ModelInfoResponse | null
  health: HealthResponse | null
  healthError: ApiError | null
}) {
  const [state, dispatch] = useReducer(wizardReducer, initialWizardState)
  const [values, setValues] = useState<FormValues>({})
  const [unavailable, setUnavailable] = useState<Record<string, boolean>>({})
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<RiskPredictionResponse | null>(null)
  const [apiError, setApiError] = useState<ApiError | null>(null)

  const { step } = state
  const serviceDown = Boolean(healthError) || health?.status === 'degraded'

  /**
   * Client-side checks mirroring the API's constraints, so obvious mistakes are
   * caught before a round trip. The API re-validates everything regardless;
   * these checks are a convenience, never the authority.
   *
   * Returns a map of API field name to message, keyed the same way the API
   * names its fields.
   */
  function validate(next: FormValues): Record<string, string> {
    const found: Record<string, string> = {}

    const age = readNumber(next, 'age')
    if (age === null) {
      found.age = 'Enter an age in years.'
    } else if (!Number.isInteger(age) || age < 1 || age > 120) {
      found.age = 'Enter a whole number between 1 and 120.'
    }

    for (const key of REQUIRED_API_KEYS) {
      if (isBlank(next[key])) found[key] = 'This field is required.'
    }

    // Every other numeric field shares one range check driven by the field
    // registry, so the form cannot drift from the bounds declared in fields.tsx.
    for (const field of FIELDS) {
      if (field.control !== 'number') continue
      if (field.apiKey === 'age') continue
      if (unavailable[field.apiKey]) continue
      const parsed = readNumber(next, field.apiKey)
      if (parsed === null) continue
      const message = numberRangeError(field, parsed)
      if (message) found[field.apiKey] = message
    }

    return found
  }

  function update(key: string, value: string | boolean | null) {
    setValues((prev) => ({ ...prev, [key]: value }))
    setErrors((prev) => {
      if (!prev[key]) return prev
      const next = { ...prev }
      delete next[key]
      return next
    })
  }

  function markUnavailable(field: FieldDef, flag: boolean) {
    setUnavailable((prev) => ({ ...prev, [field.apiKey]: flag }))
    // Marking a field unavailable clears any value it held, so nothing stale is
    // ever sent. Nothing is substituted in its place.
    if (flag) update(field.apiKey, null)
  }

  function goToReview(event: React.FormEvent) {
    event.preventDefault()
    const nextValues = { ...values }
    for (const field of Object.keys(unavailable) as string[]) {
      if (unavailable[field]) nextValues[field] = null
    }

    const problems = validate(nextValues)
    const absent = missingRequired(nextValues)
    for (const key of absent) {
      if (!problems[key]) problems[key] = 'This field is required.'
    }

    if (Object.keys(problems).length > 0) {
      setErrors(problems)
      const firstKey = Object.keys(problems)[0]
      const target = document.querySelector<HTMLElement>(`[name="${firstKey}"]`)
      target?.focus()
      target?.scrollIntoView({ block: 'center', behavior: 'smooth' })
      return
    }

    setErrors({})
    dispatch({ type: 'setValues', values: nextValues })
    dispatch({ type: 'review' })
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  async function runPrediction() {
    setSubmitting(true)
    setApiError(null)
    try {
      const payload = toRequest(state.values)
      const response = await predict(payload)
      setResult(response)
      dispatch({ type: 'result' })
    } catch (error) {
      if (error instanceof ApiError) {
        setApiError(error)
        // A 422 means the API rejected something we sent. Map it back onto the
        // form so the person sees which field, rather than a bare failure.
        if (error.kind === 'validation') {
          const mapped: Record<string, string> = {}
          for (const issue of error.fieldIssues) {
            if (issue.field) mapped[issue.field] = issue.message
          }
          if (Object.keys(mapped).length > 0) {
            setErrors(mapped)
            dispatch({ type: 'edit' })
          }
        }
      } else {
        setApiError(new ApiError('server', 0, 'Unexpected response.'))
      }
    } finally {
      setSubmitting(false)
      window.scrollTo({ top: 0, behavior: 'smooth' })
    }
  }

  function restart() {
    setValues({})
    setUnavailable({})
    setErrors({})
    setResult(null)
    setApiError(null)
    dispatch({ type: 'restart' })
  }

  // Focus management so keyboard and screen-reader users follow step changes.
  useEffect(() => {
    const heading = document.querySelector<HTMLElement>('[data-step-heading]')
    heading?.focus()
  }, [step])

  const stepIndex = STEP_INDEX[step]

  return (
    <div className="mx-auto max-w-4xl px-5 py-12 sm:px-8 sm:py-16">
      {/* Progress: a plain numbered sequence, readable without colour. */}
      <nav aria-label="Assessment progress" className="mb-12">
        <ol className="flex flex-wrap items-center gap-x-3 gap-y-2">
          {STEP_LABELS.map((label, index) => {
            const current = index === stepIndex
            const done = index < stepIndex
            return (
              <li key={label} className="flex items-center gap-3">
                <span
                  aria-current={current ? 'step' : undefined}
                  className={`inline-flex items-center gap-2 text-sm ${
                    current
                      ? 'font-semibold text-ink-900'
                      : done
                        ? 'text-ink-600'
                        : 'text-ink-400'
                  }`}
                >
                  <span
                    aria-hidden="true"
                    className={`inline-flex size-6 items-center justify-center rounded-full border text-xs tabular-nums ${
                      current
                        ? 'border-brand-600 bg-brand-600 text-white'
                        : done
                          ? 'border-ink-400 text-ink-700'
                          : 'border-ink-300'
                    }`}
                  >
                    {index + 1}
                  </span>
                  {label}
                </span>
                {index < STEP_LABELS.length - 1 ? (
                  <span aria-hidden="true" className="text-ink-300">
                    /
                  </span>
                ) : null}
              </li>
            )
          })}
        </ol>
      </nav>

      {serviceDown ? (
        <div className="mb-10">
          <ErrorNotice
            error={
              healthError ??
              new ApiError(
                'unavailable',
                503,
                'The prediction model is currently unavailable. Please retry later.',
              )
            }
          />
        </div>
      ) : null}

      {step === 'form' ? (
        <form onSubmit={goToReview} noValidate>
          <h1
            data-step-heading
            tabIndex={-1}
            className="text-[2rem] font-semibold tracking-[-0.02em] text-ink-900 focus:outline-none sm:text-[2.25rem]"
          >
            Cardiovascular risk assessment
          </h1>
          <p className="mt-3 max-w-[58ch] text-[1.0625rem] leading-relaxed text-ink-600">
            Thirteen structured readings, each with a plain-English explanation
            and the clinical term kept alongside it. Age, sex and the type of
            chest discomfort are required; everything else can be marked
            unavailable, and the model accounts for the gap rather than guessing.
          </p>

          {errors && Object.keys(errors).length > 0 ? (
            <div
              role="alert"
              className="mt-8 rounded-lg border border-alert-100 bg-alert-50 px-5 py-4"
            >
              <p className="text-sm font-medium text-alert-700">
                {Object.keys(errors).length}{' '}
                {Object.keys(errors).length === 1 ? 'field needs' : 'fields need'}{' '}
                attention before continuing.
              </p>
            </div>
          ) : null}

          <div className="mt-12 space-y-14">
            {FIELD_GROUPS.map((group) => {
              const fields = fieldsInGroup(group.id)
              if (fields.length === 0) return null
              return (
                <fieldset key={group.id} className="min-w-0">
                  <legend className="w-full">
                    <span className="rule block w-full" aria-hidden="true" />
                    <span className="mt-6 block text-lg font-semibold tracking-[-0.01em] text-ink-900">
                      {group.title}
                    </span>
                  </legend>
                  <p className="mt-1.5 max-w-[58ch] text-sm leading-relaxed text-ink-500">
                    {group.blurb}
                  </p>

                  <div className="mt-7 grid gap-7 sm:grid-cols-2 sm:gap-x-10">
                    {fields.map((field) => {
                      const isUnavailable = Boolean(unavailable[field.apiKey])
                      const raw = values[field.apiKey] ?? null
                      const error = errors[field.apiKey]

                      if (field.control === 'boolean') {
                        return (
                          <div key={field.apiKey} className="sm:col-span-2">
                            <BooleanField
                              field={field}
                              value={typeof raw === 'boolean' ? raw : null}
                              onChange={(value) => update(field.apiKey, value)}
                              error={error}
                            />
                          </div>
                        )
                      }

                      if (field.control === 'enum') {
                        return (
                          <EnumField
                            key={field.apiKey}
                            field={field}
                            value={typeof raw === 'string' ? raw : ''}
                            unavailable={isUnavailable}
                            onChange={(value) => update(field.apiKey, value)}
                            onUnavailableChange={(flag) =>
                              markUnavailable(field, flag)
                            }
                            error={error}
                          />
                        )
                      }

                      return (
                        <NumberField
                          key={field.apiKey}
                          field={field}
                          value={typeof raw === 'string' ? raw : ''}
                          unavailable={isUnavailable}
                          onChange={(value) => update(field.apiKey, value)}
                          onUnavailableChange={(flag) => markUnavailable(field, flag)}
                          error={error}
                        />
                      )
                    })}
                  </div>
                </fieldset>
              )
            })}
          </div>

          <div className="rule mt-14 flex flex-col-reverse gap-3 pt-8 sm:flex-row sm:justify-end">
            <Button variant="secondary" type="button" onClick={() => setValues({})}>
              Clear entries
            </Button>
            <Button type="submit" size="lg" disabled={serviceDown}>
              Continue to review
              <ArrowRight aria-hidden="true" className="size-4" strokeWidth={2.25} />
            </Button>
          </div>
        </form>
      ) : null}

      {step === 'review' ? (
        <div>
          <h1
            data-step-heading
            tabIndex={-1}
            className="sr-only focus:not-sr-only focus:outline-none"
          >
            Review your entries
          </h1>
          {apiError ? (
            <div className="mb-8">
              <ErrorNotice error={apiError} onRetry={runPrediction} />
            </div>
          ) : null}
          <ReviewSummary
            rows={toReviewRows(state.values)}
            onEdit={() => {
              dispatch({ type: 'edit' })
              window.scrollTo({ top: 0, behavior: 'smooth' })
            }}
            onSubmit={runPrediction}
            submitting={submitting}
          />
          {submitting ? (
            <div className="mt-6">
              <PendingNote />
            </div>
          ) : null}
        </div>
      ) : null}

      {step === 'result' && result ? (
        <RiskResult result={result} modelInfo={modelInfo} onRestart={restart} />
      ) : null}

      {step === 'result' && !result ? (
        <div className="py-12 text-center">
          <Info aria-hidden="true" className="mx-auto size-6 text-ink-400" />
          <p className="mt-4 text-ink-600">No result to show yet.</p>
          <Button className="mt-6" onClick={restart} type="button">
            Start an assessment
          </Button>
        </div>
      ) : null}

      {step === 'review' ? (
        <p className="mt-10 text-xs leading-relaxed text-ink-500">
          <ArrowLeft aria-hidden="true" className="mr-1 inline size-3" />
          Nothing is sent until you choose Run Assessment.
        </p>
      ) : null}
    </div>
  )
}
