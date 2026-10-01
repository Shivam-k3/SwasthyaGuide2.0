import type { ReactNode } from 'react'

/**
 * A single flat panel with one border and no shadow. Used sparingly, to group
 * something that genuinely needs grouping, rather than around every block.
 */
export function Panel({
  children,
  className = '',
  tone = 'light',
}: {
  children: ReactNode
  className?: string
  tone?: 'light' | 'dark'
}) {
  const base =
    tone === 'dark'
      ? 'border-ink-700 bg-ink-900 text-ink-100'
      : 'border-ink-200 bg-white text-ink-800'
  return <div className={`rounded-xl border ${base} ${className}`}>{children}</div>
}

/**
 * The small labelled figure used for real, measured numbers. Deliberately not
 * a card: it is text on the page with a rule above it.
 */
export function Stat({
  label,
  value,
  detail,
  tone = 'light',
}: {
  label: string
  value: string
  detail?: string
  tone?: 'light' | 'dark'
}) {
  const labelColor = tone === 'dark' ? 'text-ink-300' : 'text-ink-500'
  const valueColor = tone === 'dark' ? 'text-white' : 'text-ink-900'
  const detailColor = tone === 'dark' ? 'text-ink-400' : 'text-ink-500'
  return (
    <div className="border-t border-ink-200 pt-3 dark:border-ink-700">
      <dt className={`text-xs font-medium uppercase tracking-wider ${labelColor}`}>
        {label}
      </dt>
      <dd className={`mt-1.5 text-xl font-semibold tabular-nums ${valueColor}`}>
        {value}
      </dd>
      {detail ? (
        <p className={`mt-1 text-sm leading-snug ${detailColor}`}>{detail}</p>
      ) : null}
    </div>
  )
}

/** A plain horizontal meter. Used for one measured proportion, nothing else. */
export function Meter({
  value,
  label,
  tone = 'brand',
}: {
  /** A proportion in [0, 1] supplied by the backend. Not derived here. */
  value: number
  label: string
  tone?: 'brand' | 'alert'
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100
  const bar = tone === 'alert' ? 'bg-alert-600' : 'bg-brand-500'
  return (
    <div
      role="img"
      aria-label={label}
      className="h-1.5 w-full overflow-hidden rounded-full bg-ink-200"
    >
      <div className={`h-full rounded-full ${bar}`} style={{ width: `${pct}%` }} />
    </div>
  )
}
