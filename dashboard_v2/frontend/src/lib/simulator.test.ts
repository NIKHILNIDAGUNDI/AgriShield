import { describe, expect, it } from 'vitest'
import type { RegionRecord } from '../types'
import { buildScenario, defaultControls, featureSources, scenarioTargetMonth } from './simulator'

function record(year_month: string, offset: number): RegionRecord {
  return {
    region: 'US',
    year_month,
    year: 2025,
    month: Number(year_month.slice(5)),
    temperature: -3 + offset,
    precipitation: 36 + offset,
    crop_yield: 101 + offset,
    fuel_price: 67 + offset,
    transport_cost: 1.3 + offset / 100,
    inflation_rate: 2.5 + offset / 10,
    geopolitical_risk: 0.05,
    news_sentiment: 0.4,
    food_price_index: 111 + offset,
    shortage_risk_score: 5 + offset,
  }
}

const latest = record('2025-12', 0)
// Deliberately out of order, with an older month that must not be used.
const records = [latest, record('2025-10', 2), record('2025-09', 9), record('2025-11', 1)]
const unchanged = defaultControls(latest)

describe('buildScenario', () => {
  it('uses three consecutive real months when nothing is changed', () => {
    const { features, estimate } = buildScenario(records, unchanged)
    const expected = [latest, record('2025-11', 1), record('2025-10', 2)]
    expect(Object.keys(features)).toHaveLength(30)
    for (const source of featureSources) {
      expected.forEach((row, index) => {
        expect(features[`${source}_lag_${index + 1}`]).toBeCloseTo(row[source], 10)
      })
    }
    expect(estimate.price).toBeCloseTo(latest.food_price_index, 10)
    expect(estimate.risk).toBeCloseTo(latest.shortage_risk_score, 10)
  })

  it('applies weather shifts to the latest reading, not to zero or a constant', () => {
    const { features } = buildScenario(records, { ...unchanged, temperatureShift: 2, precipitationShift: -25 })
    expect(features.temperature_lag_1).toBeCloseTo(-1)
    expect(features.precipitation_lag_1).toBeCloseTo(27)
    expect(features.temperature_lag_2).toBe(-2)
    expect(features.precipitation_lag_3).toBe(38)
    expect(features.crop_yield_lag_1).toBe(101)
  })

  it('never produces negative precipitation', () => {
    const { features } = buildScenario(records, { ...unchanged, precipitationShift: -100 })
    expect(features.precipitation_lag_1).toBe(0)
  })

  it('reduces crop yield under a drought-sized shift', () => {
    for (const change of [{ temperatureShift: 3 }, { precipitationShift: -41 }]) {
      const { features, estimate } = buildScenario(records, { ...unchanged, ...change })
      expect(features.crop_yield_lag_1).toBeCloseTo(101 * 0.65)
      expect(estimate.price).toBeGreaterThan(latest.food_price_index)
    }
  })

  it('moves transport cost with fuel price and geopolitical risk', () => {
    const { features } = buildScenario(records, { ...unchanged, fuel: 117, geopolitical: 0.25 })
    expect(features.transport_cost_lag_1).toBeCloseTo(1.3 + 0.8 + 0.1)
  })

  it('does not modify the source records', () => {
    const before = JSON.stringify(records)
    buildScenario(records, { ...unchanged, temperatureShift: 5 })
    expect(JSON.stringify(records)).toBe(before)
  })

  it('targets one month past the March outlook, across year ends', () => {
    expect(scenarioTargetMonth('2025-12')).toBe('2026-04')
    expect(scenarioTargetMonth('2025-08')).toBe('2025-12')
  })

  it('rejects a history that is too short', () => {
    expect(() => buildScenario(records.slice(0, 2), unchanged)).toThrow('three months')
  })
})
