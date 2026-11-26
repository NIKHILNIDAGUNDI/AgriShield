import type { ApiModels, Forecast, RegionRecord } from '../types'

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) throw new Error(`API returned HTTP ${response.status}`)
  return response.json() as Promise<T>
}

export function fetchHealth() {
  return fetchJson<{ status: string; active_model_version: string }>('/v1/health')
}

export function fetchModels() {
  return fetchJson<ApiModels>('/v1/models')
}

export async function fetchDataset(): Promise<RegionRecord[]> {
  const response = await fetch('/food_security_data.csv')
  if (!response.ok) throw new Error(`Dataset returned HTTP ${response.status}`)
  const text = await response.text()
  const [headerLine, ...lines] = text.trim().split(/\r?\n/)
  const headers = headerLine.split(',')
  return lines.filter(Boolean).map((line) => {
    const values = line.split(',')
    const record = Object.fromEntries(headers.map((header, index) => {
      const value = values[index]
      const number = Number(value)
      return [header, value === '' ? null : (Number.isFinite(number) && header !== 'year_month' && header !== 'region' ? number : value)]
    }))
    return record as unknown as RegionRecord
  })
}

export function predictRows(rows: RegionRecord[], forecastOnly = true) {
  return fetchJson<{ predictions: Forecast[] }>('/v1/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ rows, forecast_only: forecastOnly }),
  })
}

export function predictFeatures(region: string, features: Record<string, number>) {
  return fetchJson<{ predictions: Forecast[] }>('/v1/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ region, features }),
  })
}
