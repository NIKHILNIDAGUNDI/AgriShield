import type { RegionRecord } from '../types'

export const featureSources = [
  'temperature',
  'precipitation',
  'crop_yield',
  'fuel_price',
  'transport_cost',
  'inflation_rate',
  'geopolitical_risk',
  'news_sentiment',
  'food_price_index',
  'shortage_risk_score',
] as const

export type ScenarioControls = {
  temperatureShift: number
  precipitationShift: number
  inflation: number
  fuel: number
  geopolitical: number
  sentiment: number
}

export type Scenario = {
  features: Record<string, number>
  /** Rule-of-thumb result, shown only when the model API is unavailable. */
  estimate: { price: number; risk: number }
}

/**
 * The model's features are the three months before a "feature month", and it predicts
 * three months after that. The scenario's newest input is the latest reading, so the
 * target is four months after it: one month past the last dashboard forecast.
 */
export function scenarioTargetMonth(latestMonth: string): string {
  const [year, month] = latestMonth.split('-').map(Number)
  return new Date(Date.UTC(year, month - 1 + 4, 1)).toISOString().slice(0, 7)
}

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

/** Unchanged controls: no weather shift, and the latest reading for everything else. */
export function defaultControls(latest: RegionRecord): ScenarioControls {
  return {
    temperatureShift: 0,
    precipitationShift: 0,
    inflation: latest.inflation_rate,
    fuel: latest.fuel_price,
    geopolitical: latest.geopolitical_risk,
    sentiment: latest.news_sentiment,
  }
}

// The target formulas used by the data notebook, without its noise terms.
function formulaTargets(state: RegionRecord) {
  const price = 100
    + (100 - state.crop_yield) * 1.2
    + (state.transport_cost - 1) * 45
    + state.inflation_rate * 1.5
    + state.geopolitical_risk * 30
    - state.news_sentiment * 15
  const droughtIndex = clamp((state.temperature - 15) / 5 + (100 - state.precipitation) / 20, 0, 10)
  const risk = (100 - state.crop_yield) * 0.4
    + (state.transport_cost - 1) * 20
    + state.geopolitical_risk * 25
    - (10 - droughtIndex) * 2
  return { price, risk }
}

function applyControls(latest: RegionRecord, controls: ScenarioControls): RegionRecord {
  // Only a change made by the user triggers drought; the baseline already reflects past ones.
  const drought = controls.precipitationShift < -40 || controls.temperatureShift > 2.5
  const state: RegionRecord = {
    ...latest,
    temperature: latest.temperature + controls.temperatureShift,
    precipitation: Math.max(0, latest.precipitation * (1 + controls.precipitationShift / 100)),
    crop_yield: latest.crop_yield * (drought ? 0.65 : 1),
    fuel_price: controls.fuel,
    transport_cost: latest.transport_cost
      + 0.8 * (controls.fuel - latest.fuel_price) / 50
      + 0.5 * (controls.geopolitical - latest.geopolitical_risk),
    inflation_rate: controls.inflation,
    geopolitical_risk: controls.geopolitical,
    news_sentiment: controls.sentiment,
  }
  // Move the real price and risk by the change the formulas imply, so no change means no movement.
  const before = formulaTargets(latest)
  const after = formulaTargets(state)
  state.food_price_index = clamp(latest.food_price_index + after.price - before.price, 50, 300)
  state.shortage_risk_score = clamp(latest.shortage_risk_score + after.risk - before.risk, 0, 100)
  return state
}

/**
 * Lag 1 is the latest month with the scenario applied; lags 2 and 3 are the two
 * real months before it. `records` must all belong to one region.
 */
export function buildScenario(records: RegionRecord[], controls: ScenarioControls): Scenario {
  const history = [...records].sort((a, b) => a.year_month.localeCompare(b.year_month))
  if (history.length < 3) throw new Error('At least three months of history are required.')
  const [third, second, latest] = history.slice(-3)
  const first = applyControls(latest, controls)

  const features: Record<string, number> = {}
  for (const source of featureSources) {
    features[`${source}_lag_1`] = first[source]
    features[`${source}_lag_2`] = second[source]
    features[`${source}_lag_3`] = third[source]
  }
  return {
    features,
    estimate: { price: first.food_price_index, risk: first.shortage_risk_score },
  }
}
