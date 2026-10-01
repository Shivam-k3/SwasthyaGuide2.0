import { createContext, useContext, useEffect, useState } from 'react'
import { Navigate, Route, Routes } from 'react-router-dom'
import { SiteHeader } from './components/layout/SiteHeader'
import { SiteFooter } from './components/layout/SiteFooter'
import { ApiError, getHealth, getModelInfo } from './api/client'
import type { HealthResponse, ModelInfoResponse } from './api/types'
import { Home } from './pages/Home'
import { Assessment } from './pages/Assessment'
import { HowItWorks } from './pages/HowItWorks'
import { About } from './pages/About'

/**
 * Service metadata is fetched once at start-up and shared, because
 * /model-info does not change while the process runs. A failure here degrades
 * the page rather than breaking it.
 */
interface ServiceState {
  health: HealthResponse | null
  healthError: ApiError | null
  modelInfo: ModelInfoResponse | null
}

const ServiceContext = createContext<ServiceState>({
  health: null,
  healthError: null,
  modelInfo: null,
})

export function useService() {
  return useContext(ServiceContext)
}

export function App() {
  const [state, setState] = useState<ServiceState>({
    health: null,
    healthError: null,
    modelInfo: null,
  })

  useEffect(() => {
    let active = true

    getHealth()
      .then((health) => {
        if (active) setState((prev) => ({ ...prev, health, healthError: null }))
      })
      .catch((error: unknown) => {
        if (active && error instanceof ApiError) {
          setState((prev) => ({ ...prev, healthError: error }))
        }
      })

    getModelInfo()
      .then((modelInfo) => {
        if (active) setState((prev) => ({ ...prev, modelInfo }))
      })
      .catch(() => {
        /* The pages degrade to static copy when this is unavailable. */
      })

    return () => {
      active = false
    }
  }, [])

  const healthSummary = state.health
    ? { model: state.health.model, deployment_threshold: state.health.deployment_threshold }
    : null

  return (
    <ServiceContext.Provider value={state}>
      <div className="flex min-h-dvh flex-col">
        <a
          href="#main"
          className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-white focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-ink-900 focus:shadow-lg focus:outline focus:outline-2 focus:outline-brand-600"
        >
          Skip to main content
        </a>
        <SiteHeader />
        <main id="main" className="flex-1">
          <Routes>
            <Route
              path="/"
              element={
                <Home modelInfo={state.modelInfo} health={healthSummary} />
              }
            />
            <Route
              path="/assessment"
              element={
                <Assessment
                  modelInfo={state.modelInfo}
                  health={state.health}
                  healthError={state.healthError}
                />
              }
            />
            <Route path="/how-it-works" element={<HowItWorks />} />
            <Route path="/about" element={<About />} />
            {/* Unknown routes redirect home, as specified. */}
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
        <SiteFooter />
      </div>
    </ServiceContext.Provider>
  )
}
