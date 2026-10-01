import { Check, Pencil } from 'lucide-react'
import { Button } from '../ui/Button'
import type { ReviewRow } from '../../state/wizard'

/**
 * Read-only confirmation of everything that will be sent.
 *
 * Values are shown exactly as entered, including the fields marked unavailable,
 * so nothing is substituted without the person seeing it first.
 */
export function ReviewSummary({
  rows,
  onEdit,
  onSubmit,
  submitting,
}: {
  rows: ReviewRow[]
  onEdit: () => void
  onSubmit: () => void
  submitting: boolean
}) {
  return (
    <div className="mx-auto max-w-3xl">
      <h2 className="text-2xl font-semibold tracking-[-0.015em] text-ink-900">
        Review your entries
      </h2>
      <p className="mt-3 text-[1.0625rem] leading-relaxed text-ink-600">
        Nothing has been sent yet. Check the readings below, then run the
        assessment. Fields marked not available are sent as missing, and the
        model accounts for them explicitly.
      </p>

      <dl className="mt-10">
        {rows.map((row) => (
          <div
            key={row.field.apiKey}
            className="flex items-baseline justify-between gap-6 border-t border-ink-200 py-3.5"
          >
            <dt className="text-[0.9375rem] text-ink-600">{row.field.label}</dt>
            <dd
              className={`text-right text-[0.9375rem] tabular-nums ${
                row.available ? 'font-medium text-ink-900' : 'text-ink-400'
              }`}
            >
              {row.display}
            </dd>
          </div>
        ))}
      </dl>

      <div className="rule mt-8 flex flex-col-reverse gap-3 pt-8 sm:flex-row sm:justify-end">
        <Button variant="secondary" onClick={onEdit} type="button">
          <Pencil aria-hidden="true" className="size-4" strokeWidth={2} />
          Edit
        </Button>
        <Button onClick={onSubmit} type="button" disabled={submitting} size="lg">
          {submitting ? 'Running assessment' : 'Run Assessment'}
          {!submitting ? (
            <Check aria-hidden="true" className="size-4" strokeWidth={2.25} />
          ) : null}
        </Button>
      </div>
    </div>
  )
}
