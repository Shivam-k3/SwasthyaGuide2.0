import { useEffect, useRef, useState } from 'react'
import {
  ArrowDown,
  BarChart3,
  FileText,
  Lock,
  Microscope,
  SlidersHorizontal,
} from 'lucide-react'
import { HeartHero } from '../components/layout/HeartHero'
import { ButtonLink } from '../components/ui/Button'
import { SectionHeading } from '../components/ui/Section'
import { Stat } from '../components/ui/Panel'
import type { ModelInfoResponse } from '../api/types'

/** Thresholds are only ever read from the backend. Nothing is computed here. */
function Findings({
  modelInfo,
  health,
}: {
  modelInfo: ModelInfoResponse | null
  health: { model: string; deployment_threshold: number } | null
}) {
  return (
    <>
      <div className="grid gap-x-10 gap-y-6 sm:grid-cols-3">
        <Stat
          label="Model served"
          value={health?.model ?? modelInfo?.model ?? 'Logistic regression'}
          detail={
            health
              ? 'Loaded from the persisted Stage 3 pipeline'
              : 'Reading from the service'
          }
        />
        <Stat
          label="Operating threshold"
          value={
            health
              ? `${(health.deployment_threshold * 100).toFixed(0)}%`
              : 'Not connected'
          }
          detail="Fixed on the server. Not adjustable from this page."
        />
        <Stat
          label="Clinical inputs"
          value={String(modelInfo?.n_input_features ?? 13)}
          detail="Structured readings, each of which can be marked unavailable"
        />
      </div>
    </>
  )
}

/** Scrolls the next block into view on load, which anchors a long landing page. */
function useScrollHint() {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    const node = ref.current
    if (!node) return
    const observer = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { threshold: 0.6 },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [])
  return { ref, visible }
}

export function Home({
  modelInfo,
  health,
}: {
  modelInfo: ModelInfoResponse | null
  health: { model: string; deployment_threshold: number } | null
}) {
  const { ref, visible } = useScrollHint()

  return (
    <>
      <HeartHero />

      <section
        ref={ref}
        aria-hidden="true"
        className={`flex justify-center bg-ink-50 pt-14 transition-opacity duration-500 ${
          visible ? 'opacity-100' : 'opacity-0'
        }`}
      >
        <ArrowDown className="size-5 text-ink-400" />
      </section>

      {/* What it does. Deliberately not three equal cards: a lede, then a
          measured figure strip, then the detail in an asymmetric two-column. */}
      <section className="mx-auto max-w-6xl px-5 py-20 sm:px-8 sm:py-24">
        <div className="grid gap-12 lg:grid-cols-[minmax(0,0.8fr)_minmax(0,1.2fr)] lg:gap-16">
          <SectionHeading
            eyebrow="What this does"
            title="A risk score with its own uncertainty attached"
            lede="Most risk tools hand you a percentage and nothing else. This one returns the percentage together with how the model actually behaved on held-back development data at the same threshold it used for your result."
          />

          <div className="space-y-10 text-[1.0625rem] leading-relaxed text-ink-700">
            <p>
              The pipeline behind SWASTHYAGUIDE was developed and frozen in an
              earlier stage of this project. It has never been retrained by the
              web interface, and it does not learn from anything you type.
            </p>
            <p>
              You supply the readings a clinician would already have: age, sex,
              chest pain type, blood pressure, cholesterol, maximum heart rate,
              exercise response and a few test results. Anything you do not have
              can be marked unavailable, and the pipeline handles the gap
              explicitly rather than guessing a value.
            </p>
            <p>
              The result reports a model-estimated risk score and one of two
              classifications.
              It does not claim that you have, or do not have, heart disease.
              That judgement is not available from this data.
            </p>
          </div>
        </div>

        <div className="mt-16">
          <Findings modelInfo={modelInfo} health={health} />
        </div>
      </section>

      {/* Capabilities. An uneven four-item list rather than a repeated card row:
          alternating alignment gives the eye somewhere to land. */}
      <section className="border-y border-ink-200 bg-white">
        <div className="mx-auto max-w-6xl px-5 py-20 sm:px-8 sm:py-24">
          <SectionHeading
            eyebrow="Capabilities"
            title="What the interface guarantees"
            lede="Four properties that follow from how this system was built, rather than from its interface."
          />

          <dl className="mt-14 space-y-0">
            {[
              {
                icon: Lock,
                title: 'The score is computed on the server',
                body: 'Every probability, threshold and classification you see is returned by the inference API. This interface performs no medical scoring of its own, and exposes no control that would change the operating point.',
              },
              {
                icon: BarChart3,
                title: 'The model reports its own error rates',
                body: 'Recall, specificity, and the false-positive and false-negative rates come from the same Stage 3 record the model was selected on. They describe the model on development data, not your personal outlook.',
              },
              {
                icon: SlidersHorizontal,
                title: 'Missing data stays missing',
                body: 'A field marked unavailable is sent as absent. Nothing is silently replaced with a zero, an average, or any other substitute, so the model always knows what it does not know.',
              },
              {
                icon: Microscope,
                title: 'Research status is stated plainly',
                body: 'This is a prototype built on 920 rows from a public research dataset. It has no external validation and has not been assessed for clinical use.',
              },
            ].map((item, index) => {
              const Icon = item.icon
              const right = index % 2 === 1
              return (
                <div
                  key={item.title}
                  className={`grid gap-4 border-t border-ink-200 py-9 sm:grid-cols-[auto_minmax(0,1fr)] sm:gap-8 ${
                    right ? 'sm:pl-[18%]' : 'sm:pr-[18%]'
                  }`}
                >
                  <Icon
                    aria-hidden="true"
                    className="mt-0.5 size-5 shrink-0 text-brand-600"
                    strokeWidth={1.75}
                  />
                  <div className={right ? 'sm:text-right' : ''}>
                    <dt className="text-lg font-semibold tracking-[-0.01em] text-ink-900">
                      {item.title}
                    </dt>
                    <dd
                      className={`mt-2 max-w-[62ch] leading-relaxed text-ink-600 ${
                        right ? 'sm:ml-auto' : ''
                      }`}
                    >
                      {item.body}
                    </dd>
                  </div>
                </div>
              )
            })}
          </dl>
        </div>
      </section>

      {/* Closing. Deep field to echo the hero, with one clear next step. */}
      <section className="bg-ink-900">
        <div className="mx-auto grid max-w-6xl gap-10 px-5 py-20 sm:px-8 sm:py-24 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-end">
          <div className="max-w-[40rem]">
            <SectionHeading
              tone="onDark"
              eyebrow="Next step"
              title="Run an assessment in about two minutes"
              lede="Thirteen readings, most of which you may already know. Anything missing can be marked unavailable and the model will account for it."
            />
          </div>
          <div className="flex flex-col gap-3 sm:flex-row lg:flex-col">
            <ButtonLink to="/assessment" size="lg">
              <FileText aria-hidden="true" className="size-4" strokeWidth={2.25} />
              Start Assessment
            </ButtonLink>
            <ButtonLink to="/how-it-works" size="lg" variant="quiet" className="!text-ink-200 hover:!bg-ink-800 hover:!text-white">
              How It Works
            </ButtonLink>
          </div>
        </div>
      </section>
    </>
  )
}
