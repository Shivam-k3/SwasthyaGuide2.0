import { Activity } from 'lucide-react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { ButtonLink } from '../ui/Button'

const LINKS = [
  { to: '/', label: 'Home', end: true },
  { to: '/assessment', label: 'Assessment', end: false },
  { to: '/how-it-works', label: 'How It Works', end: false },
  { to: '/about', label: 'About', end: false },
] as const

function Wordmark() {
  return (
    <Link
      to="/"
      className="flex shrink-0 items-baseline gap-1.5 rounded text-ink-900 transition-opacity hover:opacity-80"
    >
      <span className="text-[0.9375rem] font-semibold tracking-[-0.01em]">
        SWASTHYAGUIDE
      </span>
      <span className="text-[0.8125rem] font-medium text-brand-700">2.0</span>
    </Link>
  )
}

export function SiteHeader() {
  const [open, setOpen] = useState(false)
  const location = useLocation()

  useEffect(() => {
    setOpen(false)
  }, [location.pathname])

  return (
    <header className="sticky top-0 z-40 border-b border-ink-200 bg-ink-50/90 backdrop-blur-sm">
      <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-4 px-5 sm:px-8">
        <Wordmark />

        <nav aria-label="Primary" className="hidden items-center gap-1 md:flex">
          {LINKS.map((link) => (
            <NavLink
              key={link.to}
              to={link.to}
              end={link.end}
              className={({ isActive }) =>
                `rounded-lg px-3 py-2 text-[0.9375rem] transition-colors ${
                  isActive
                    ? 'font-medium text-ink-900'
                    : 'text-ink-600 hover:text-ink-900'
                }`
              }
            >
              {link.label}
            </NavLink>
          ))}
        </nav>

        <div className="flex items-center gap-2">
          <ButtonLink
            to="/assessment"
            size="md"
            className="hidden sm:inline-flex"
          >
            <Activity aria-hidden="true" className="size-4" strokeWidth={2.25} />
            Start Assessment
          </ButtonLink>

          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            aria-expanded={open}
            aria-controls="mobile-nav"
            className="inline-flex min-h-11 min-w-11 items-center justify-center rounded-lg border border-ink-300 text-ink-700 md:hidden"
          >
            <span className="sr-only">
              {open ? 'Close menu' : 'Open menu'}
            </span>
            <svg
              aria-hidden="true"
              viewBox="0 0 20 20"
              className="size-5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.75"
              strokeLinecap="round"
            >
              {open ? (
                <path d="M5 5l10 10M15 5L5 15" />
              ) : (
                <path d="M3 6h14M3 10h14M3 14h14" />
              )}
            </svg>
          </button>
        </div>
      </div>

      {open ? (
        <nav
          id="mobile-nav"
          aria-label="Primary"
          className="border-t border-ink-200 bg-white md:hidden"
        >
          <ul className="mx-auto max-w-6xl px-5 py-2 sm:px-8">
            {LINKS.map((link) => (
              <li key={link.to}>
                <NavLink
                  to={link.to}
                  end={link.end}
                  className={({ isActive }) =>
                    `block rounded-lg px-3 py-3 text-base ${
                      isActive
                        ? 'font-medium text-ink-900'
                        : 'text-ink-600'
                    }`
                  }
                >
                  {link.label}
                </NavLink>
              </li>
            ))}
            <li className="px-3 pb-3 pt-1">
              <ButtonLink to="/assessment" full>
                <Activity aria-hidden="true" className="size-4" strokeWidth={2.25} />
                Start Assessment
              </ButtonLink>
            </li>
          </ul>
        </nav>
      ) : null}
    </header>
  )
}
