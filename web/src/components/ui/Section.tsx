import type { ReactNode } from 'react'

interface EyebrowProps {
  children: ReactNode
  tone?: 'default' | 'onDark'
  className?: string
}

/**
 * A small label that sets the scale for the section that follows. Used instead
 * of a large heading, so sections stay subordinate to the page title.
 */
export function Eyebrow({ children, tone = 'default', className = '' }: EyebrowProps) {
  const color = tone === 'onDark' ? 'text-brand-200' : 'text-brand-700'
  return (
    <p
      className={`text-xs font-semibold uppercase tracking-[0.14em] ${color} ${className}`}
    >
      {children}
    </p>
  )
}

interface SectionHeadingProps {
  eyebrow?: string
  title: string
  lede?: ReactNode
  tone?: 'default' | 'onDark'
  className?: string
  /** Optional heading level, so a page keeps a single correct h1. */
  as?: 'h1' | 'h2' | 'h3'
}

export function SectionHeading({
  eyebrow,
  title,
  lede,
  tone = 'default',
  className = '',
  as: Heading = 'h2',
}: SectionHeadingProps) {
  const headingColor = tone === 'onDark' ? 'text-white' : 'text-ink-900'
  const ledeColor =
    tone === 'onDark' ? 'text-ink-200' : 'text-ink-600'
  return (
    <div className={className}>
      {eyebrow ? <Eyebrow tone={tone}>{eyebrow}</Eyebrow> : null}
      <Heading
        className={`mt-2.5 text-2xl font-semibold tracking-[-0.015em] sm:text-[1.75rem] sm:leading-[1.2] ${headingColor}`}
      >
        {title}
      </Heading>
      {lede ? (
        <p className={`mt-3 max-w-[54ch] text-[1.0625rem] leading-relaxed ${ledeColor}`}>
          {lede}
        </p>
      ) : null}
    </div>
  )
}
