import { Link } from 'react-router-dom'
import { ButtonLink } from '../components/ui/Button'
import { SectionHeading } from '../components/ui/Section'

const STAGES = [
  {
    number: '01',
    title: 'You enter structured cardiovascular information',
    body: 'Thirteen readings a clinician would already have taken: demographics, chest pain type, blood pressure, cholesterol, maximum heart rate, exercise response, and any test results available. Anything you do not have can be marked unavailable.',
  },
  {
    number: '02',
    title: 'The data is processed by the frozen pipeline',
    body: 'The frozen logistic regression pipeline handles imputation, missingness indicators, encoding and scaling exactly as it was fitted during model development. The web interface performs none of this work itself.',
  },
  {
    number: '03',
    title: 'The model returns a probability and a classification',
    body: 'Inference produces one predicted risk figure and one classification against a fixed operating point. The threshold is set on the server and is not adjustable from the interface.',
  },
  {
    number: '04',
    title: 'The result is presented with its context and limits',
    body: 'Alongside your score, the service reports how that model actually behaved on held-back development data at the same threshold, including how often it raises a false alarm and how often it misses a case.',
  },
]

const LIMITATIONS = [
  '920 rows from a public research dataset, which is small by clinical standards.',
  'Four cohorts that differ in how they were recruited and are therefore not directly comparable.',
  'No external validation on any population outside the original dataset.',
  'Three candidate models that repeated cross-validation could not separate on discrimination once tuned.',
  'A locked holdout set of 184 rows, used once and never for tuning or threshold selection.',
]

export function HowItWorks() {
  return (
    <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8 sm:py-20">
      <div className="max-w-[52rem]">
        <SectionHeading
          as="h1"
          eyebrow="Method"
          title="How a result is produced"
          lede="Four stages, each of which happens somewhere other than where you might expect. The model was built and frozen in an earlier stage of this project; this interface only reads from it."
        />
      </div>

      <ol className="mt-16 max-w-[52rem]">
        {STAGES.map((stage) => (
          <li
            key={stage.number}
            className="grid grid-cols-[2.5rem_minmax(0,1fr)] gap-x-5 border-t border-ink-200 py-8"
          >
            <span
              aria-hidden="true"
              className="pt-1 text-sm font-semibold tabular-nums text-brand-600"
            >
              {stage.number}
            </span>
            <div>
              <h2 className="text-lg font-semibold tracking-[-0.01em] text-ink-900">
                {stage.title}
              </h2>
              <p className="mt-2.5 leading-relaxed text-ink-600">{stage.body}</p>
            </div>
          </li>
        ))}
      </ol>

      {/* The false-alarm principle, given its own block because it is the single
          most important thing to understand about the output. */}
      <section className="mt-16 rounded-xl border border-ink-200 bg-white px-6 py-8 sm:px-9 sm:py-10">
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] lg:gap-14">
          <div>
            <SectionHeading
              eyebrow="Design principle"
              title="A false alarm is preferred to a missed alarm"
            />
            <p className="mt-4 leading-relaxed text-ink-600">
              For a screening tool, telling someone they need further assessment
              when they do not is a recoverable inconvenience. Failing to flag
              someone who does need it is not. The model is therefore operated at
              a deliberately recall-oriented threshold, which accepts a higher
              rate of unnecessary flags in exchange for missing fewer cases.
            </p>
          </div>

          <div className="lg:pl-8">
            <h3 className="text-sm font-semibold uppercase tracking-[0.12em] text-ink-500">
              Why there is no threshold control
            </h3>
            <p className="mt-3 leading-relaxed text-ink-600">
              The operating point is chosen once, from measured development
              data, and held on the server. Exposing a slider would let anyone
              move the system away from the trade-off it was validated under,
              without changing any of the reported error rates to match. Every
              figure you are shown comes from the same measured record.
            </p>
            <p className="mt-4 leading-relaxed text-ink-600">
              The result page always states the recall, specificity, false
              positive rate and false negative rate for your result, so the cost
              of this choice is visible rather than assumed.
            </p>
          </div>
        </div>
      </section>

      {/* Limitations, given real weight rather than a footnote. */}
      <section className="mt-16">
        <div className="max-w-[52rem]">
          <SectionHeading
            eyebrow="Limitations"
            title="What this system cannot tell you"
            lede="These are the constraints of the underlying research, and none of them is resolved by a better interface."
          />
        </div>

        <ul className="mt-10 grid max-w-[64rem] gap-x-12 gap-y-4 sm:grid-cols-2">
          {LIMITATIONS.map((item) => (
            <li key={item} className="flex gap-3 leading-relaxed text-ink-600">
              <span
                aria-hidden="true"
                className="mt-2.5 size-1.5 shrink-0 rounded-full bg-ink-400"
              />
              <span>{item}</span>
            </li>
          ))}
        </ul>
      </section>

      <div className="rule mt-16 flex flex-col gap-4 pt-10 sm:flex-row sm:items-center">
        <p className="max-w-[46ch] text-[0.9375rem] leading-relaxed text-ink-600">
          See the full project background and research status on the about page.
        </p>
        <div className="flex flex-col gap-3 sm:ml-auto sm:flex-row">
          <ButtonLink to="/about" variant="secondary">
            About the project
          </ButtonLink>
          <ButtonLink to="/assessment">Start Assessment</ButtonLink>
        </div>
      </div>

      <p className="mt-8 text-sm text-ink-500">
        <Link
          to="/about"
          className="rounded underline underline-offset-4 hover:text-ink-800"
        >
          Read about this prototype
        </Link>{' '}
        before relying on any result it produces.
      </p>
    </div>
  )
}
