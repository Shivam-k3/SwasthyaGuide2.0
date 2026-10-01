import type { FieldDef } from '../../config/fields'

/**
 * The shared two-layer label for every assessment field.
 *
 * Layer 1 (primary): a plain-English `label`, plus the unit where relevant.
 * Layer 2 (secondary): the clinical term, kept visible but visually quiet so a
 * research or professional user can identify the exact variable being collected.
 *
 * Nothing here changes what is sent to the API; that is always `field.apiKey`.
 */
export function FieldLabel({
  field,
  required = false,
}: {
  field: FieldDef
  required?: boolean
}) {
  return (
    <span className="text-sm font-medium text-ink-800">
      {field.label}
      {field.unit ? (
        <span className="ml-1.5 font-normal text-ink-500">({field.unit})</span>
      ) : null}
      {required ? (
        <span className="ml-1.5 text-xs font-normal text-ink-500">required</span>
      ) : null}
    </span>
  )
}

export function FieldGuide({
  id,
  field,
  unavailable = false,
}: {
  id?: string
  field: FieldDef
  unavailable?: boolean
}) {
  const prefix = field.clinicalTermPrefix ?? 'Clinical term'
  return (
    <div id={id} className="mt-1 space-y-1">
      <p className="text-sm leading-relaxed text-ink-500">
        {unavailable ? 'Not recorded, or not available.' : field.description}
      </p>
      {field.unavailableNote && !unavailable ? (
        <p className="text-xs leading-relaxed text-ink-500">
          {field.unavailableNote}
        </p>
      ) : null}
      {field.clinicalTerm ? (
        <p className="text-xs leading-relaxed text-ink-500">
          {prefix}:{' '}
          <span className="font-medium text-ink-600">{field.clinicalTerm}</span>
        </p>
      ) : null}
    </div>
  )
}
