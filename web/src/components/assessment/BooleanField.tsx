import { useId } from 'react'
import type { FieldDef } from '../../config/fields'
import { FieldGuide, FieldLabel } from './FieldCopy'

interface BooleanFieldProps {
  field: FieldDef
  value: boolean | null
  onChange: (value: boolean | null) => void
  error?: string
}

/**
 * A three-way control for a yes/no reading: yes, no, or not available.
 *
 * A two-state toggle would force a false answer, because absence is meaningful
 * here and the pipeline treats it differently from a recorded "no".
 */
export function BooleanField({ field, value, onChange, error }: BooleanFieldProps) {
  const groupId = useId()
  const helpId = `${groupId}-help`
  const errorId = `${groupId}-error`

  const choices: { key: 'yes' | 'no' | 'na'; label: string; checked: boolean }[] = [
    { key: 'yes', label: 'Yes', checked: value === true },
    { key: 'no', label: 'No', checked: value === false },
    { key: 'na', label: 'Not available', checked: value === null },
  ]

  return (
    <fieldset
      className="flex flex-col"
      aria-describedby={[helpId, error ? errorId : undefined]
        .filter(Boolean)
        .join(' ')}
      aria-invalid={error ? true : undefined}
    >
      <legend>
        <FieldLabel field={field} />
      </legend>
      <FieldGuide id={helpId} field={field} />

      <div className="mt-2.5 flex flex-wrap gap-2">
        {choices.map((choice) => (
          <label
            key={choice.key}
            className={`inline-flex min-h-11 cursor-pointer select-none items-center rounded-lg border px-3.5 py-2 text-sm transition-colors ${
              choice.checked
                ? 'border-brand-600 bg-brand-50 font-medium text-brand-700'
                : 'border-ink-300 bg-white text-ink-700 hover:border-ink-400'
            }`}
          >
            <input
              type="radio"
              name={field.apiKey}
              checked={choice.checked}
              onChange={() => onChange(choice.key === 'na' ? null : choice.key === 'yes')}
              className="sr-only"
            />
            {choice.label}
          </label>
        ))}
      </div>

      {error ? (
        <p id={errorId} className="mt-2 text-sm font-medium text-alert-700">
          {error}
        </p>
      ) : null}
    </fieldset>
  )
}
