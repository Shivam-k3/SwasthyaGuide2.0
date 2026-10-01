import { afterEach, describe, expect, it, vi } from 'vitest'
import { ApiError, getHealth, predict } from '../api/client'

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  })
}

describe('api client', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('returns parsed data on 200', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(200, {
          status: 'ok',
          api_stage: '4',
          api_version: '4.0.0',
          model_loaded: true,
          model: 'logistic_regression',
          deployment_threshold: 0.4,
          disclaimer: 'x',
        }),
      ),
    )
    const health = await getHealth()
    expect(health.model).toBe('logistic_regression')
    expect(health.deployment_threshold).toBe(0.4)
  })

  it('maps a 422 into field-keyed issues and does not throw a raw body', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(422, {
          detail: [
            {
              type: 'literal_error',
              loc: ['body', 'cp'],
              msg: "Input should be 'asymptomatic'...",
            },
            { type: 'less_than_equal', loc: ['body', 'ca'], msg: 'less than 3' },
          ],
        }),
      ),
    )

    await expect(
      predict({ age: 50, sex: 'Male', cp: 'typical angina' }),
    ).rejects.toMatchObject({
      kind: 'validation',
      status: 422,
    })

    try {
      await predict({ age: 50, sex: 'Male', cp: 'typical angina' })
    } catch (error) {
      const apiError = error as ApiError
      const fields = apiError.fieldIssues.map((i) => i.field).sort()
      expect(fields).toEqual(['ca', 'cp'])
    }
  })

  it('surfaces the backend 503 message verbatim', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(503, {
          detail: 'The prediction model is currently unavailable. Please retry later.',
        }),
      ),
    )

    await expect(
      predict({ age: 50, sex: 'Male', cp: 'typical angina' }),
    ).rejects.toMatchObject({
      kind: 'unavailable',
      status: 503,
      userMessage:
        'The prediction model is currently unavailable. Please retry later.',
    })
  })

  it('turns a network failure into a clear message with no stack trace', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new TypeError('Failed to fetch')
      }),
    )

    try {
      await getHealth()
      throw new Error('should have thrown')
    } catch (error) {
      const apiError = error as ApiError
      expect(apiError.kind).toBe('network')
      expect(apiError.userMessage).toBe('Cannot reach the assessment service.')
      expect(apiError.userMessage).not.toContain('TypeError')
      expect(apiError.userMessage).not.toContain('Failed to fetch')
    }
  })

  it('never surfaces a raw 500 body, which could contain a traceback', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () =>
        jsonResponse(500, {
          detail:
            'Traceback (most recent call last): File "C:\\app\\model_service.py", line 42',
        }),
      ),
    )

    try {
      await predict({ age: 50, sex: 'Male', cp: 'typical angina' })
      throw new Error('should have thrown')
    } catch (error) {
      const apiError = error as ApiError
      expect(apiError.kind).toBe('server')
      expect(apiError.userMessage).not.toContain('Traceback')
      expect(apiError.userMessage).not.toContain('model_service.py')
      expect(apiError.userMessage).not.toContain('C:\\')
    }
  })
})
