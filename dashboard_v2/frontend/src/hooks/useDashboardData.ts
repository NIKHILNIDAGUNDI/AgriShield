import { useCallback, useEffect, useState } from 'react'
import { fetchDataset, fetchHealth, fetchMetrics, fetchModels, predictRows } from '../lib/api'
import { createCustomRegion } from '../lib/customRegion'
import { staticFallback } from '../lib/forecast'
import type { Forecast, RegionRecord, TrainingMetrics } from '../types'

export function useDashboardData() {
  const [records, setRecords] = useState<RegionRecord[]>([])
  const [forecasts, setForecasts] = useState<Forecast[]>([])
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null)
  const [modelVersion, setModelVersion] = useState<string | null>(null)
  const [apiAvailable, setApiAvailable] = useState(false)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState('')

  useEffect(() => {
    let cancelled = false
    async function initialize() {
      try {
        const dataset = await fetchDataset()
        if (cancelled) return
        setRecords(dataset)
        setLoading(false)
        void fetchMetrics().then((result) => { if (!cancelled) setMetrics(result) }).catch(() => {})

        const [health, models] = await Promise.allSettled([fetchHealth(), fetchModels()])
        if (cancelled) return
        setModelVersion(
          (models.status === 'fulfilled' ? models.value.active_model_version : null)
          ?? (health.status === 'fulfilled' ? health.value.active_model_version : null),
        )

        try {
          const prediction = await predictRows(dataset, true)
          if (cancelled) return
          setForecasts(prediction.predictions)
          setApiAvailable(true)
        } catch {
          if (cancelled) return
          setApiAvailable(false)
          setForecasts(staticFallback(dataset))
        }
      } catch (error) {
        setLoading(false)
        setLoadError(error instanceof Error ? error.message : 'Dashboard data could not be loaded.')
      }
    }
    void initialize()
    return () => { cancelled = true }
  }, [])

  const markApiUnavailable = useCallback(() => setApiAvailable(false), [])

  /** Looks up a new region, forecasts it, and returns its resolved name. */
  async function addRegion(name: string, onStatus: (status: string) => void) {
    const result = await createCustomRegion(name)
    onStatus(`Fetched ${result.region}. Calibrating forecast...`)
    let predictions: Forecast[]
    try {
      predictions = (await predictRows(result.records, true)).predictions
      setApiAvailable(true)
    } catch {
      setApiAvailable(false)
      predictions = staticFallback(result.records)
    }
    setRecords((current) => [...current.filter((record) => record.region !== result.region), ...result.records])
    setForecasts((current) => [...current.filter((item) => item.region !== result.region), ...predictions])
    return result.region
  }

  return { records, forecasts, metrics, modelVersion, apiAvailable, loading, loadError, addRegion, markApiUnavailable }
}
