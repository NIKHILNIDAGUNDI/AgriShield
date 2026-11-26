export type RegionRecord = {
  year_month: string
  year: number
  month: number
  crop_yield: number
  fuel_price: number
  transport_cost: number
  inflation_rate: number
  geopolitical_risk: number
  news_sentiment: number
  food_price_index: number
  shortage_risk_score: number
  temperature: number
  precipitation: number
  region: string
  data_provenance?: Record<string, string>
}

export type Forecast = {
  region: string
  feature_month: string
  target_month: string
  predicted_price: number
  predicted_shortage_risk: number
  price_interval?: { lower: number; upper: number }
  shortage_risk_interval?: { lower: number; upper: number }
}

export type ApiModels = {
  active_model_version: string
  models: Array<{ version: string; active: boolean; verification: string }>
}
