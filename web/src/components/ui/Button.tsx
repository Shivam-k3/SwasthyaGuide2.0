import type { ButtonHTMLAttributes, ReactNode } from 'react'
import { Link } from 'react-router-dom'

type Variant = 'primary' | 'secondary' | 'quiet' | 'danger'
type Size = 'md' | 'lg'

const BASE =
  'inline-flex items-center justify-center gap-2 rounded-lg font-medium ' +
  'transition-colors duration-150 disabled:cursor-not-allowed ' +
  'disabled:opacity-55 min-h-11'

const SIZES: Record<Size, string> = {
  md: 'px-4 py-2.5 text-[0.9375rem]',
  lg: 'px-5 py-3 text-base',
}

const VARIANTS: Record<Variant, string> = {
  primary: 'bg-brand-600 text-white hover:bg-brand-700 active:bg-brand-700',
  secondary:
    'bg-white text-ink-800 border border-ink-300 hover:border-ink-400 hover:bg-ink-50',
  quiet: 'text-ink-700 hover:text-ink-900 hover:bg-ink-100/70',
  danger: 'bg-alert-600 text-white hover:bg-alert-700',
}

interface Common {
  variant?: Variant
  size?: Size
  full?: boolean
  children: ReactNode
  className?: string
}

export function Button({
  variant = 'primary',
  size = 'md',
  full,
  className = '',
  children,
  ...rest
}: Common & ButtonHTMLAttributes<HTMLButtonElement>) {
  return (
    <button
      className={`${BASE} ${SIZES[size]} ${VARIANTS[variant]} ${
        full ? 'w-full' : ''
      } ${className}`}
      {...rest}
    >
      {children}
    </button>
  )
}

export function ButtonLink({
  to,
  variant = 'primary',
  size = 'md',
  full,
  className = '',
  children,
}: Common & { to: string }) {
  return (
    <Link
      to={to}
      className={`${BASE} ${SIZES[size]} ${VARIANTS[variant]} ${
        full ? 'w-full' : ''
      } ${className}`}
    >
      {children}
    </Link>
  )
}
