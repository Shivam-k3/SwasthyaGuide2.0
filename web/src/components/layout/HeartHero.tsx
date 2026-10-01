import { Link } from 'react-router-dom'

const ECG_PATH =
  'M0 32 L52 32 L60 32 L66 14 L74 50 L82 32 L96 32 L104 24 L110 40 L118 32 L188 32'

/**
 * The landing hero.
 *
 * Asymmetric by intent: copy occupies a narrower measure on the left, the heart
 * carries the right half on desktop and moves below the copy on mobile. The
 * image is the identity, so the panel behind it is a plain deep-navy field
 * rather than a card.
 */
export function HeartHero() {
  return (
    <section className="relative overflow-hidden bg-ink-900">
      {/* Two soft washes, kept off the palette's extremes so the heart stays the
          brightest thing in the frame. */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-40 right-[-10%] size-[46rem] rounded-full bg-brand-600/22 blur-3xl"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute bottom-[-30%] left-[-15%] size-[36rem] rounded-full bg-brand-500/10 blur-3xl"
      />

      <div className="relative mx-auto grid max-w-6xl gap-12 px-5 pb-16 pt-14 sm:px-8 sm:pt-20 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)] lg:items-center lg:gap-10 lg:pb-24 lg:pt-24">
        <div className="max-w-[34rem] animate-rise">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-brand-300">
            Research prototype
          </p>

          <h1 className="mt-5 text-[2.125rem] font-semibold leading-[1.12] tracking-[-0.022em] text-white sm:text-[2.75rem] lg:text-[3.125rem]">
            Understand your cardiovascular risk from clinical readings
          </h1>

          <p className="mt-6 text-[1.0625rem] leading-relaxed text-ink-200">
            Enter thirteen structured clinical readings. A logistic regression
            pipeline that was already trained and frozen returns a research risk
            score, together with the measured behaviour of that model at the
            operating point it uses.
          </p>

          <div className="mt-9 flex flex-col gap-3 sm:flex-row sm:items-center">
            <Link
              to="/assessment"
              className="inline-flex min-h-11 items-center justify-center rounded-lg bg-brand-500 px-5 py-3 text-base font-medium text-white transition-colors hover:bg-brand-600"
            >
              Start Assessment
            </Link>
            <Link
              to="/how-it-works"
              className="inline-flex min-h-11 items-center justify-center rounded-lg px-5 py-3 text-base font-medium text-ink-200 transition-colors hover:text-white"
            >
              How It Works
            </Link>
          </div>

          <p className="mt-8 max-w-[46ch] border-l-2 border-ink-700 pl-4 text-sm leading-relaxed text-ink-300">
            This is a research risk score, not a diagnosis. It does not rule out
            any condition and cannot replace a clinician's assessment.
          </p>
        </div>

        <div className="relative animate-rise-delay lg:pl-2">
          <div className="relative mx-auto max-w-[34rem]">
            <img
              src="/assets/heart-hero.png"
              alt="Illustration of a human heart with an electrocardiogram trace running through it."
              width={1536}
              height={1024}
              className="w-full select-none rounded-2xl object-cover"
              fetchPriority="high"
              decoding="async"
            />

            {/* ECG trace echo, decorative and hidden from assistive technology. */}
            <svg
              aria-hidden="true"
              viewBox="0 0 188 64"
              preserveAspectRatio="none"
              className="pointer-events-none absolute inset-x-0 bottom-0 h-14 w-full opacity-60"
            >
              <path
                d={ECG_PATH}
                fill="none"
                stroke="#5cc0c2"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="ecg-trace"
              />
            </svg>
          </div>

          <p className="mt-4 text-center text-xs text-ink-400">
            Logistic regression served at a recall-oriented operating point.
            Illustrative image.
          </p>
        </div>
      </div>
    </section>
  )
}
