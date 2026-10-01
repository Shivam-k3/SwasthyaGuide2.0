/**
 * The explicit "not available" control, shared by every optional field.
 *
 * This is a real checkbox rather than an empty input so a person who does not
 * know the value has a visible, deliberate way to say so. It is not a zero and
 * not a guess: marking a field unavailable sends `null`, which the model
 * imputes from training data with an explicit missingness indicator.
 */
export function UnavailableToggle({
  inputId,
  checked,
  onChange,
}: {
  inputId: string
  checked: boolean
  onChange: (checked: boolean) => void
}) {
  return (
    <label
      htmlFor={inputId}
      className="flex min-h-11 cursor-pointer select-none items-center gap-2 text-sm text-ink-700"
    >
      <input
        id={inputId}
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(event.target.checked)}
        className="size-[1.125rem] shrink-0 cursor-pointer rounded border-ink-400 accent-brand-600"
      />
      Not available
    </label>
  )
}

/** Shared input border styling, so a flagged field looks the same everywhere. */
export function fieldClassName(hasError: boolean): string {
  return `min-h-11 w-full rounded-lg border bg-white px-3 py-2 text-base text-ink-900 tabular-nums transition-colors placeholder:text-ink-400 disabled:cursor-not-allowed disabled:bg-ink-100 disabled:text-ink-400 ${
    hasError ? 'border-alert-600' : 'border-ink-300 hover:border-ink-400'
  }`
}

/** The per-field validation message, rendered identically by every field type. */
export function FieldError({ id, message }: { id: string; message?: string }) {
  if (!message) return null
  return (
    <p id={id} className="mt-2 text-sm font-medium text-alert-700">
      {message}
    </p>
  )
}
