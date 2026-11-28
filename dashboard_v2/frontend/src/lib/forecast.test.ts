import { describe, expect, it } from 'vitest'
import type { RegionRecord } from '../types'
import { staticFallback, targetMonths } from './forecast'

const record = (month: number, price: number, risk: number) => ({
  region: 'US',
  year_month: `2025-${month}`,
  food_price_index: price,
  shortage_risk_score: risk,
}) as RegionRecord

describe('staticFallback', () => {
  it('extends the recent trend and keeps values in range', () => {
    const result = staticFallback([record(12, 120, 98), record(10, 100, 90), record(11, 110, 94)])
    expect(result.map((item) => item.target_month)).toEqual(targetMonths)
    expect(result.map((item) => item.predicted_price)).toEqual([130, 140, 150])
    expect(result.map((item) => item.predicted_shortage_risk)).toEqual([100, 100, 100])
  })
})
