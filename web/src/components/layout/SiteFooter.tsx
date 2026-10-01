import { Link } from 'react-router-dom'

export function SiteFooter() {
  return (
    <footer className="mt-24 border-t border-ink-200 bg-white">
      <div className="mx-auto max-w-6xl px-5 py-12 sm:px-8">
        <div className="grid gap-10 sm:grid-cols-[1.4fr_1fr_1fr]">
          <div>
            <p className="text-[0.9375rem] font-semibold text-ink-900">
              SWASTHYAGUIDE <span className="text-brand-700">2.0</span>
            </p>
            <p className="mt-2.5 max-w-[42ch] text-sm leading-relaxed text-ink-600">
              A research prototype for cardiovascular risk assessment. It
              produces a research risk score and is not a diagnostic tool.
            </p>
          </div>

          <nav aria-label="Footer">
            <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-ink-500">
              Pages
            </h2>
            <ul className="mt-3 space-y-2">
              {[
                { to: '/', label: 'Home' },
                { to: '/assessment', label: 'Assessment' },
                { to: '/how-it-works', label: 'How It Works' },
                { to: '/about', label: 'About' },
              ].map((link) => (
                <li key={link.to}>
                  <Link
                    to={link.to}
                    className="rounded text-sm text-ink-600 transition-colors hover:text-ink-900"
                  >
                    {link.label}
                  </Link>
                </li>
              ))}
            </ul>
          </nav>

          <div>
            <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-ink-500">
              Before you rely on this
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-ink-600">
              This output must not be used to diagnose, treat, or rule out any
              condition. Consult a qualified healthcare professional.
            </p>
          </div>
        </div>

        <p className="rule mt-10 pt-6 text-xs text-ink-500">
          Research prototype. Not a medical device and not validated for clinical
          decision-making.
        </p>
      </div>
    </footer>
  )
}
