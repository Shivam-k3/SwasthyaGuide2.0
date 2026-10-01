import type {
  ApiValidationError,
  HealthResponse,
  ModelInfoResponse,
  PatientRiskRequest,
  RiskPredictionResponse,
} from './types'

const CONFIGURED_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '')

/**
 * In development the app always speaks to its own origin and lets the Vite
 * proxy forward /api/* to FastAPI. The Stage 4 server sends no CORS headers, so
 * a cross-origin call would be blocked; the proxy keeps the request
 * same-origin and no CORS middleware is needed.
 *
 * In production VITE_API_BASE_URL may name the API origin. When it is unset the
 * app falls back to same-origin relative paths.
 */
export const apiBaseUrl = import.meta.env.DEV ? '' : CONFIGURED_BASE

const API_PREFIX = import.meta.env.DEV ? '/api' : ''

export type ApiErrorKind = 'validation' | 'unavailable' | 'network' | 'server'

export interface FieldIssue {
  /** API field name, e.g. "age". Empty when the error is not field-specific. */
  field: string
  message: string
}

export class ApiError extends Error {
  readonly kind: ApiErrorKind
  readonly status: number
  readonly fieldIssues: FieldIssue[]
  /** Copy that is safe to show a user. Never a stack trace or a file path. */
  readonly userMessage: string

  constructor(
    kind: ApiErrorKind,
    status: number,
    userMessage: string,
    fieldIssues: FieldIssue[] = [],
  ) {
    super(userMessage)
    this.name = 'ApiError'
    this.kind = kind
    this.status = status
    this.userMessage = userMessage
    this.fieldIssues = fieldIssues
  }
}

const UNAVAILABLE_COPY =
  'The prediction model is currently unavailable. Please retry later.'

function url(path: string): string {
  return `${apiBaseUrl}${API_PREFIX}${path}`
}

/**
 * Map FastAPI's 422 body onto form fields. `loc` is ["body", field, ...], so
 * the field name is the second element when present.
 */
function readValidationIssues(detail: unknown): FieldIssue[] {
  if (!Array.isArray(detail)) return []
  const issues: FieldIssue[] = []
  for (const entry of detail as ApiValidationError[]) {
    const loc = entry?.loc ?? []
    const fieldIndex = loc[0] === 'body' ? 1 : 0
    const raw = loc[fieldIndex]
    issues.push({
      field: typeof raw === 'string' ? raw : '',
      message: entry?.msg ?? 'This value is not valid.',
    })
  }
  return issues
}

/**
 * Never surface a raw backend body for an unexpected failure: a 500 here could
 * in principle contain a traceback or a filesystem path, which does not belong
 * in the interface.
 */
function messageForStatus(status: number): string {
  if (status === 503) return UNAVAILABLE_COPY
  if (status === 422) return 'Some entries need attention before this can run.'
  return 'The assessment service returned an unexpected response. Please try again.'
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(url(path), {
      headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
      ...init,
    })
  } catch {
    throw new ApiError('network', 0, 'Cannot reach the assessment service.')
  }

  if (response.ok) {
    return (await response.json()) as T
  }

  let detail: unknown
  try {
    const body: unknown = await response.json()
    detail = (body as { detail?: unknown })?.detail
  } catch {
    detail = undefined
  }

  if (response.status === 422) {
    const fieldIssues = readValidationIssues(detail)
    throw new ApiError(
      'validation',
      422,
      messageForStatus(422),
      fieldIssues,
    )
  }

  if (response.status === 503) {
    // The backend already returns a deliberately generic message. Prefer it.
    const copy =
      typeof detail === 'string' && detail.trim() ? detail : UNAVAILABLE_COPY
    throw new ApiError('unavailable', 503, copy)
  }

  throw new ApiError('server', response.status, messageForStatus(response.status))
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health')
}

export function getModelInfo(): Promise<ModelInfoResponse> {
  return request<ModelInfoResponse>('/model-info')
}

export function predict(body: PatientRiskRequest): Promise<RiskPredictionResponse> {
  return request<RiskPredictionResponse>('/predict', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}
