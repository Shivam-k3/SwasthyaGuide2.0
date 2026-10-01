import { useId } from 'react'
import type { FieldDef } from '../../config/fields'
import {
  FieldError,
  UnavailableToggle,
  fieldClassName,
} from './FieldControls'
import { FieldGuide, FieldLabel } from './FieldCopy'

interface EnumFieldProps {
  field: FieldDef
  value: string
  unavailable: boolean
  onChange: (value: string) => void
  onUnavailableChange: (unavailable: boolean) => void
  error?: string
}

/**
 * A select for categorical readings.
 *
 * `option.value` is the backend's own string and is sent unchanged. Only
 * `option.label` is presentation, so a person reads "Non-anginal pain" while the
 * API still receives "non-anginal".
 */
export function EnumField({
  field,
  value,
  unavailable,
  onChange,
  onUnavailableChange,
  error,
}: EnumFieldProps) {
  const selectId = useId()
  const helpId = `${selectId}-help`
  const errorId = `${selectId}-error`
  const naId = `${selectId}-na`
  const required = Boolean(field.required)

  const describedBy = [helpId, error ? errorId : undefined].filter(Boolean).join(' ')

  return (
    <div className="flex flex-col">
      <label htmlFor={selectId}>
        <FieldLabel field={field} required={required} />
      </label>

      <FieldGuide id={helpId} field={field} unavailable={unavailable} />

      <div className="mt-2.5 flex items-center gap-4">
        <select
          id={selectId}
          name={field.apiKey}
          value={unavailable ? '' : value}
          onChange={(event) => onChange(event.target.value)}
          disabled={unavailable}
          required={required}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy || undefined}
          className={`${fieldClassName(Boolean(error))} max-w-[17rem]`}
        >
          <option value="" disabled>
            Select one
          </option>
          {field.options?.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>

        {/* Required fields have no missing state, so they get no toggle. */}
        {!required ? (
          <UnavailableToggle
            inputId={naId}
            checked={unavailable}
            onChange={onUnavailableChange}
          />
        ) : null}
      </div>

      <FieldError id={errorId} message={error} />
    </div>
  )
}
