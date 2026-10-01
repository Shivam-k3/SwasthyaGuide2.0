import { ButtonLink } from '../components/ui/Button'
import { SectionHeading } from '../components/ui/Section'

export function About() {
  return (
    <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8 sm:py-20">
      <div className="max-w-[50rem]">
        <SectionHeading
          as="h1"
          eyebrow="About"
          title="SWASTHYAGUIDE 2.0"
          lede="A research prototype for cardiovascular risk assessment, built on a public research dataset and a modelling pipeline that was developed, evaluated and then frozen before this interface existed."
        />
      </div>

      <div className="mt-14 grid gap-14 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:gap-20">
        <section>
          <h2 className="text-lg font-semibold tracking-[-0.01em] text-ink-900">
            Purpose
          </h2>
          <div className="mt-4 space-y-4 leading-relaxed text-ink-600">
            <p>
              The project set out to answer a narrow question: given thirteen
              routine cardiovascular readings, how much can a small, transparent,
              model say about estimated risk, and what would it cost you in
              false alarms to say it?
            </p>
            <p>
              That second half of the question shaped everything. A model tuned
              for headline accuracy can be made to look better by accepting more
              missed cases, so the operating point was fixed deliberately at a
              recall-oriented threshold and the cost reported alongside every
              result.
            </p>
            <p>
              The delivered system is an interface over that frozen pipeline. It
              computes no risk of its own and holds no clinical logic.
            </p>
          </div>
        </section>

        <section>
          <h2 className="text-lg font-semibold tracking-[-0.01em] text-ink-900">
            Methodology context
          </h2>
          <div className="mt-4 space-y-4 leading-relaxed text-ink-600">
            <p>
              The underlying data is a 920-row research dataset assembled from
              four cohorts. Because those cohorts were collected under different
              protocols, they are not directly comparable, and performance is
              reported per cohort as well as overall.
            </p>
            <p>
              Model selection used repeated stratified cross-validation on the
              training portion only, with per-fold results retained so that
              differences between candidate models could be compared
              pair-wise rather than by headline average.
            </p>
            <p>
              A locked holdout of 184 rows was set aside at the start and
              evaluated once, at the end, after every modelling choice was
              frozen. It was never used for tuning or for choosing the operating
              threshold.
            </p>
          </div>
        </section>
      </div>

      {/* Transparency, stated as a property of the build rather than a claim. */}
      <section className="mt-16 rounded-xl border border-ink-200 bg-white px-6 py-8 sm:px-9 sm:py-10">
        <SectionHeading
          eyebrow="Transparency"
          title="What you can inspect"
          lede="Nothing about the model is hidden behind a login or a marketing page."
        />

        <dl className="mt-9 grid gap-x-12 gap-y-6 sm:grid-cols-2">
          {[
            [
              'The model itself',
              'Logistic regression, selected over two tree-based alternatives on cross-validated discrimination and calibration. Shown on every result page, including when it is not the default.',
            ],
            [
              'The operating point',
              'Its value, the measured recall and specificity at that value, and the provenance of the choice.',
            ],
            [
              'The error rates',
              'Recall, specificity, precision, F1, false-positive rate, false-negative rate, ROC-AUC and PR-AUC, with the confusion matrix behind them.',
            ],
            [
              'The source of each number',
              'Every figure carries a statement of which development set it was measured on, and confirmation that the locked holdout was not used.',
            ],
            [
              'The development data',
              'Size, cohort structure, seed and dataset checksum, published by the inference API.',
            ],
            [
              'How missing data is handled',
              'The service reports, per result, which fields were absent and how the pipeline accounted for them.',
            ],
          ].map(([term, description]) => (
            <div key={term} className="border-t border-ink-200 pt-4">
              <dt className="text-sm font-semibold text-ink-900">{term}</dt>
              <dd className="mt-1.5 text-sm leading-relaxed text-ink-600">
                {description}
              </dd>
            </div>
          ))}
        </dl>
      </section>

      <section className="mt-16 grid gap-10 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] lg:gap-16">
        <SectionHeading
          eyebrow="Status"
          title="Research prototype"
          lede="This is an academic project. It is not a medical device, has not been validated for clinical decision-making, and is not deployed for patient care."
        />

        <div className="space-y-5 leading-relaxed text-ink-600">
          <p>
            No clinical trial has been conducted. No regulatory clearance has been
            sought or granted. No patient data is collected, stored or retained
            by this application, and nothing you enter leaves your browser except
            to compute your result.
          </p>
          <p>
            The result is a research risk score derived from population-level
            patterns in historical data. It cannot account for your individual
            history, your other conditions, your medications or your current
            symptoms, all of which matter more to your actual health than
            anything on this form.
          </p>
          <p>
            If you have chest pain, shortness of breath or any other concerning
            symptom, seek medical attention rather than relying on a score from
            this page.
          </p>
        </div>
      </section>

      <section className="mt-16 rounded-xl border border-alert-100 bg-alert-50 px-6 py-8 sm:px-9">
        <h2 className="text-lg font-semibold text-alert-700">Disclaimer</h2>
        <p className="mt-3 max-w-[62ch] leading-relaxed text-ink-700">
          SWASTHYAGUIDE 2.0 is a research prototype for health risk assessment.
          It is not a medical device, not a diagnostic system, and not validated
          for clinical decision-making. Its output is a research risk score, not
          a diagnosis, and must not be used to diagnose, treat, or rule out any
          condition. Consult a qualified healthcare professional.
        </p>
      </section>

      <div className="rule mt-14 flex flex-col gap-3 pt-8 sm:flex-row">
        <ButtonLink to="/how-it-works">How it works</ButtonLink>
        <ButtonLink to="/assessment" variant="secondary">
          Start an assessment
        </ButtonLink>
      </div>
    </div>
  )
}
