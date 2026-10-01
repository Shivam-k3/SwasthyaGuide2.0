import { useState } from 'react'
import type { ModelInfoResponse, RiskPredictionResponse } from '../../api/types'
import { Button, ButtonLink } from '../ui/Button'
import { Meter, Stat } from '../ui/Panel'
import { SectionHeading } from '../ui/Section'

const PERCENT = new Intl.NumberFormat('en-US', {
  style: 'percent',
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
})

/**
 * Formatting only. The classification, threshold and probability below all come
 * from the API response; this component decides nothing about what they mean.
 */
function formatPercent(value: number): string {
  return PERCENT.format(value)
}

function VerdictLabel({ prediction }: { prediction: string }) {
  return prediction === 'elevated_risk' ? (
    <span className="text-alert-700">Elevated predicted risk</span>
  ) : (
    <span className="text-brand-700">Lower predicted risk according to the model</span>
  )
}

export function RiskResult({
  result,
  modelInfo,
  onRestart,
}: {
  result: RiskPredictionResponse
  modelInfo: ModelInfoResponse | null
  onRestart: () => void
}) {
  const [showModel, setShowModel] = useState(false)
  const op = result.expected_operating_point
  const elevated = result.prediction === 'elevated_risk'

  return (
    <div className="mx-auto max-w-3xl">
      <SectionHeading
        eyebrow="Your model result"
        title="Model-estimated risk"
        lede="This figure comes from the assessment service. It is a research risk score produced by a model, not a measurement of your health."
      />

      {/* Primary figure. The meter is a visual restatement of the same number,
          not a second calculation, and the value is stated in the alt text. */}
      <div className="mt-10 rounded-xl border border-ink-200 bg-white p-7 sm:p-9">
        <p className="text-sm font-medium uppercase tracking-[0.12em] text-ink-500">
          Model-estimated risk
        </p>
        <p className="mt-2 text-[3.25rem] font-semibold leading-none tracking-[-0.03em] text-ink-900 tabular-nums sm:text-[4rem]">
          {formatPercent(result.risk_probability)}
        </p>

        <div className="mt-6">
          <Meter
            value={result.risk_probability}
            tone={elevated ? 'alert' : 'brand'}
            label={`Model-estimated risk ${formatPercent(result.risk_probability)}`}
          />
          <p className="mt-2.5 text-sm text-ink-500">
            Classified against a fixed operating point of{' '}
            {formatPercent(result.threshold)} set by the service. The
            classification below is returned by the API, not derived here.
          </p>
        </div>

        <div
          className={`mt-7 rounded-lg border-l-[3px] px-5 py-4 ${
            elevated
              ? 'border-alert-600 bg-alert-50'
              : 'border-brand-600 bg-brand-50'
          }`}
        >
          <p className="text-lg font-semibold">
            <VerdictLabel prediction={result.prediction} />
          </p>
          <p className="mt-2 leading-relaxed text-ink-700">{result.message}</p>
        </div>
      </div>

      {result.notes.length > 0 ? (
        <section className="mt-10">
          <h3 className="text-base font-semibold text-ink-900">
            How your entries were handled
          </h3>
          <p className="mt-1.5 text-sm text-ink-600">
            Notes returned by the model pipeline about missing or unusual values.
          </p>
          <ul className="mt-4 space-y-2.5">
            {result.notes.map((note) => (
              <li key={note} className="flex gap-3 text-[0.9375rem] text-ink-700">
                <span aria-hidden="true" className="mt-2 size-1.5 shrink-0 rounded-full bg-brand-500" />
                <span className="leading-relaxed">{note}</span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}

      {/* Explanation. Model-level context only: the API returns no per-patient
          feature contributions, so none are invented here. */}
      <section className="mt-10">
        <h3 className="text-base font-semibold text-ink-900">
          Why did the model produce this result?
        </h3>
        <p className="mt-2 max-w-[62ch] leading-relaxed text-ink-600">
          Features contributing to the model's prediction, in the sense the model
          uses them. The figures below describe how this model behaved on
          held-back development data at this operating point. They describe the
          model, not this individual, and no single feature caused this outcome.
        </p>

        <dl className="mt-6 grid gap-x-10 gap-y-6 sm:grid-cols-2">
          <Stat
            label="Recall"
            value={formatPercent(op.recall)}
            detail={`${op.tp} of ${op.n_positive} development cases found`}
          />
          <Stat
            label="Specificity"
            value={formatPercent(op.specificity)}
            detail={`${op.tn} of ${op.tn + op.fp} correctly cleared`}
          />
          <Stat
            label="False-positive rate"
            value={formatPercent(op.false_positive_rate)}
            detail={`${op.fp} unnecessary flags, which this design accepts deliberately`}
          />
          <Stat
            label="False-negative rate"
            value={formatPercent(op.false_negative_rate)}
            detail={`${op.fn} missed cases`}
          />
          <Stat
            label="Precision"
            value={formatPercent(op.precision)}
            detail="Of those flagged, how many were cases"
          />
          <Stat
            label="F1 score"
            value={formatPercent(op.f1)}
            detail="Balance of precision and recall"
          />
          <Stat
            label="ROC-AUC"
            value={op.roc_auc.toFixed(3)}
            detail="Ranking quality across all thresholds"
          />
          <Stat
            label="PR-AUC"
            value={op.pr_auc.toFixed(3)}
            detail="Ranking quality on the positive class"
          />
        </dl>

        <p className="mt-6 text-sm leading-relaxed text-ink-500">{op.source}</p>

        <div className="mt-6 rounded-lg border border-ink-200 bg-white px-5 py-4">
          <p className="text-sm font-medium text-ink-800">
            The false-alarm bias is intentional
          </p>
          <p className="mt-1.5 text-sm leading-relaxed text-ink-600">
            This system is tuned to prefer a false alarm over a missed case, so
            more people are flagged than strictly necessary. The threshold is
            fixed on the server and cannot be changed from here.
          </p>
        </div>
      </section>

      {/* Technical detail, collapsed by default so it does not crowd a person
          who came here for an answer. */}
      <section className="mt-10">
        <button
          type="button"
          onClick={() => setShowModel((v) => !v)}
          aria-expanded={showModel}
          aria-controls="model-detail"
          className="flex min-h-11 w-full items-center justify-between gap-4 rounded-lg border border-ink-200 bg-white px-5 py-3 text-left text-[0.9375rem] font-medium text-ink-900 transition-colors hover:border-ink-300"
        >
          Model Information
          <span aria-hidden="true" className="text-ink-400">
            {showModel ? 'Hide' : 'Show'}
          </span>
        </button>

        <div id="model-detail" hidden={!showModel} className="mt-4">
          <dl className="rounded-xl border border-ink-200 bg-white px-5 py-4">
            {[
              ['Model', result.model],
              ['Estimator', modelInfo?.model_type ?? 'Not reported'],
              ['Model version', result.model_version],
              ['Operating threshold', result.threshold.toFixed(2)],
              [
                'Deployment selection',
                modelInfo?.selection_rationale ?? 'Not reported',
              ],
              ['Calibration', modelInfo?.calibration_note ?? 'Not reported'],
              ['Provenance', modelInfo?.provenance ?? 'Not reported'],
              ['Threshold provenance', modelInfo?.threshold_source ?? 'Not reported'],
              ['Deviation policy', modelInfo?.threshold_policy ?? 'Not reported'],
              [
                'Development confusion matrix',
                `TP ${op.tp}, FP ${op.fp}, TN ${op.tn}, FN ${op.fn} across ${op.n_rows} rows`,
              ],
            ].map(([term, value]) => (
              <div
                key={term}
                className="flex flex-col gap-1 border-b border-ink-100 py-3 last:border-b-0 sm:flex-row sm:gap-6"
              >
                <dt className="text-sm text-ink-500 sm:w-52 sm:shrink-0">{term}</dt>
                <dd className="text-sm leading-relaxed text-ink-800">{value}</dd>
              </div>
            ))}
          </dl>
        </div>
      </section>

      <div className="rule mt-10 flex flex-col gap-6 pt-8">
        <div className="rounded-lg border border-ink-300 bg-white px-5 py-4">
          <p className="text-sm font-medium text-ink-900">Important</p>
          <p className="mt-1.5 text-sm leading-relaxed text-ink-600">
            {result.disclaimer}
          </p>
        </div>

        <div className="flex flex-col gap-3 sm:flex-row">
          <ButtonLink to="/how-it-works" variant="secondary">
            How this result was produced
          </ButtonLink>
          <Button variant="quiet" onClick={onRestart} type="button">
            Start another assessment
          </Button>
        </div>
      </div>
    </div>
  )
}
