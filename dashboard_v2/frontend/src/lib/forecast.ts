import type { Forecast, RegionRecord } from '../types'

export const targetMonths = ['2026-01', '2026-02', '2026-03']
export const outlookMonth = '2026-03'
export const heatmapMonths = [
  '2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06',
  '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12',
  ...targetMonths,
]

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

export function regionRecords(records: RegionRecord[], region: string) {
  return records
    .filter((record) => record.region === region)
    .sort((a, b) => a.year_month.localeCompare(b.year_month))
}

/** Linear extrapolation of the last three months, used when the API is unavailable. */
export function staticFallback(records: RegionRecord[]): Forecast[] {
  const regions = [...new Set(records.map((record) => record.region))]
  return regions.flatMap((region) => {
    const recent = regionRecords(records, region).slice(-3)
    const prices = recent.map((row) => row.food_price_index)
    const risks = recent.map((row) => row.shortage_risk_score)
    const priceTrend = recent.length > 1 ? (prices.at(-1)! - prices[0]) / 2 : 0
    const riskTrend = recent.length > 1 ? (risks.at(-1)! - risks[0]) / 2 : 0
    return targetMonths.map((targetMonth, index) => ({
      region,
      feature_month: recent.at(-1)?.year_month ?? '2025-12',
      target_month: targetMonth,
      predicted_price: clamp(prices.at(-1)! + priceTrend * (index + 1), 50, 300),
      predicted_shortage_risk: clamp(risks.at(-1)! + riskTrend * (index + 1), 0, 100),
    }))
  })
}
