import { Link } from 'react-router-dom'
import { ButtonLink } from '../components/ui/Button'

export function NotFound() {
  return (
    <div className="mx-auto flex min-h-[60vh] max-w-2xl flex-col justify-center px-5 py-20 sm:px-8">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-brand-700">
        404
      </p>
      <h1 className="mt-3 text-[2rem] font-semibold tracking-[-0.02em] text-ink-900">
        That page does not exist
      </h1>
      <p className="mt-4 text-[1.0625rem] leading-relaxed text-ink-600">
        The address may have been mistyped, or the page may have moved.
      </p>
      <div className="mt-9 flex flex-col gap-3 sm:flex-row">
        <ButtonLink to="/">Back to home</ButtonLink>
        <ButtonLink to="/assessment" variant="secondary">
          Start Assessment
        </ButtonLink>
      </div>
      <p className="mt-10 text-sm text-ink-500">
        <Link
          to="/how-it-works"
          className="rounded underline underline-offset-4 hover:text-ink-800"
        >
          How this result is produced
        </Link>
      </p>
    </div>
  )
}
