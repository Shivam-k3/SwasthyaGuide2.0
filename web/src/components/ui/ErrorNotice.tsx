import { AlertTriangle, Loader2, ServerCrash, WifiOff } from 'lucide-react'
import type { ApiError } from '../../api/client'

interface ErrorNoticeProps {
  error: ApiError
  onRetry?: () => void
}

const PRESENTATION: Record<
  ApiError['kind'],
  { icon: typeof WifiOff; heading: string; hint: string }
> = {
  unavailable: {
    icon: ServerCrash,
    heading: 'Assessment service unavailable',
    hint: 'The model is not loaded on the server right now. This is a temporary condition and nothing you entered has been lost.',
  },
  network: {
    icon: WifiOff,
    heading: 'Cannot reach the assessment service',
    hint: 'Check that the backend is running, then try again.',
  },
  validation: {
    icon: AlertTriangle,
    heading: 'Some entries need attention',
    hint: 'Correct the highlighted fields and run the assessment again.',
  },
  server: {
    icon: AlertTriangle,
    heading: 'Unexpected response',
    hint: 'The service did not respond as expected. Please try again.',
  },
}

/**
 * Renders only the backend's safe, generic message. A traceback or a filesystem
 * path is never surfaced, and a 5xx body is never dumped to the screen.
 */
export function ErrorNotice({ error, onRetry }: ErrorNoticeProps) {
  const presentation = PRESENTATION[error.kind]
  const Icon = presentation.icon

  return (
    <div
      role="alert"
      className="rounded-xl border border-alert-100 bg-alert-50 px-5 py-5 sm:px-6"
    >
      <div className="flex gap-3.5">
        <Icon
          aria-hidden="true"
          className="mt-0.5 size-5 shrink-0 text-alert-600"
          strokeWidth={2}
        />
        <div className="min-w-0">
          <p className="font-semibold text-alert-700">{presentation.heading}</p>
          <p className="mt-1.5 leading-relaxed text-ink-700">{error.userMessage}</p>
          <p className="mt-1.5 text-sm leading-relaxed text-ink-500">
            {presentation.hint}
          </p>
          {onRetry ? (
            <button
              type="button"
              onClick={onRetry}
              className="mt-4 min-h-11 rounded-lg border border-ink-300 bg-white px-4 py-2 text-sm font-medium text-ink-800 transition-colors hover:border-ink-400"
            >
              Try again
            </button>
          ) : null}
        </div>
      </div>
    </div>
  )
}

/** A brief inline busy note. Deliberately short: inference is not slow. */
export function PendingNote() {
  return (
    <p
      aria-live="polite"
      className="flex items-center gap-2 text-sm text-ink-600"
    >
      <Loader2 aria-hidden="true" className="size-4 animate-spin" strokeWidth={2.25} />
      Running the assessment
    </p>
  )
}
