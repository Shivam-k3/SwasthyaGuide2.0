import { useId } from 'react'
import type { FieldDef } from '../../config/fields'
import {
  FieldError,
  UnavailableToggle,
  fieldClassName,
} from './FieldControls'
import { FieldGuide, FieldLabel } from './FieldCopy'

interface NumberFieldProps {
  field: FieldDef
  value: string
  unavailable: boolean
  onChange: (value: string) => void
  onUnavailableChange: (unavailable: boolean) => void
  error?: string
}

/**
 * A numeric reading with an explicit "not available" switch.
 *
 * `oldpeak` deliberately has no `min`: negative ST depression is a legitimate
 * recorded value in this project, so the field never blocks one.
 */
export function NumberField({
  field,
  value,
  unavailable,
  onChange,
  onUnavailableChange,
  error,
}: NumberFieldProps) {
  const inputId = useId()
  const helpId = `${inputId}-help`
  const errorId = `${inputId}-error`
  const naId = `${inputId}-na`

  // The helper text is hidden while the field is marked unavailable, because the
  // guide already explains the field as "not recorded" in that state.
  const describedBy = [unavailable ? undefined : helpId, error ? errorId : undefined]
    .filter(Boolean)
    .join(' ')

  const min = field.min !== undefined ? String(field.min) : undefined
  const max = field.max !== undefined ? String(field.max) : undefined
  const step = field.step !== undefined ? String(field.step) : 'any'

  return (
    <div className="flex flex-col">
      <label htmlFor={inputId}>
        <FieldLabel field={field} />
      </label>

      <FieldGuide id={helpId} field={field} unavailable={unavailable} />

      <div className="mt-2.5 flex items-center gap-4">
        <input
          id={inputId}
          name={field.apiKey}
          type="number"
          inputMode={field.step === 1 ? 'numeric' : 'decimal'}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          disabled={unavailable}
          min={min}
          max={max}
          step={step}
          required={field.required}
          aria-invalid={error ? true : undefined}
          aria-describedby={describedBy || undefined}
          placeholder={unavailable ? 'Not available' : ''}
          className={`${fieldClassName(Boolean(error))} max-w-[11rem]`}
        />

        <UnavailableToggle
          inputId={naId}
          checked={unavailable}
          onChange={onUnavailableChange}
        />
      </div>

      <FieldError id={errorId} message={error} />
    </div>
  )
}
