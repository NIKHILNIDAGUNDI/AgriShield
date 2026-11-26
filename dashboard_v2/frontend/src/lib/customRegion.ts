import type { RegionRecord } from '../types'

const clamp = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

export async function createCustomRegion(name: string): Promise<{ region: string; records: RegionRecord[] }> {
  const normalized = name.trim()
  if (normalized.length < 2 || normalized.length > 100 || !/^[\p{L}\p{M} .'-]+$/u.test(normalized)) {
    throw new Error('Enter a valid region name between 2 and 100 characters.')
  }

  const geoResponse = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(normalized)}&count=1&language=en&format=json`)
  if (!geoResponse.ok) throw new Error(`Geocoding returned HTTP ${geoResponse.status}`)
  const geo = await geoResponse.json() as { results?: Array<{ name: string; latitude: number; longitude: number; country_code?: string }> }
  const match = geo.results?.[0]
  if (!match) throw new Error(`Could not resolve "${normalized}".`)

  const lat = Number(match.latitude)
  const lon = Number(match.longitude)
  const countryCode = String(match.country_code || 'USA').toUpperCase()
  if (!Number.isFinite(lat) || lat < -90 || lat > 90 || !Number.isFinite(lon) || lon < -180 || lon > 180) {
    throw new Error('Geocoding returned malformed coordinates.')
  }

  const weatherUrl = `https://archive-api.open-meteo.com/v1/archive?latitude=${lat}&longitude=${lon}&start_date=2021-01-01&end_date=2025-12-31&daily=temperature_2m_mean,precipitation_sum&timezone=auto`
  const weatherResponse = await fetch(weatherUrl)
  if (!weatherResponse.ok) throw new Error(`Weather returned HTTP ${weatherResponse.status}`)
  const weather = await weatherResponse.json() as {
    daily?: { time: string[]; temperature_2m_mean: Array<number | null>; precipitation_sum: Array<number | null> }
  }
  if (!weather.daily?.time?.length) throw new Error('Weather API returned malformed observations.')

  const monthly: Record<string, { temperatures: number[]; precipitation: number[] }> = {}
  weather.daily.time.forEach((date, index) => {
    const month = date.slice(0, 7)
    monthly[month] ??= { temperatures: [], precipitation: [] }
    const temperature = weather.daily?.temperature_2m_mean[index]
    const precipitation = weather.daily?.precipitation_sum[index]
    if (typeof temperature === 'number' && Number.isFinite(temperature) && temperature >= -100 && temperature <= 100) monthly[month].temperatures.push(temperature)
    if (typeof precipitation === 'number' && Number.isFinite(precipitation) && precipitation >= 0 && precipitation <= 1000) monthly[month].precipitation.push(precipitation)
  })

  const inflationMap: Record<number, number> = {}
  try {
    const inflationResponse = await fetch(`https://api.worldbank.org/v2/country/${encodeURIComponent(countryCode)}/indicator/FP.CPI.TOTL.ZG?date=2021:2025&format=json`)
    if (inflationResponse.ok) {
      const inflation = await inflationResponse.json() as Array<unknown>
      const rows = Array.isArray(inflation) && Array.isArray(inflation[1]) ? inflation[1] as Array<{ date?: string; value?: number | null }> : []
      rows.forEach((row) => {
        const year = Number(row.date)
        const value = row.value
        if (Number.isInteger(year) && year >= 2021 && year <= 2025 && typeof value === 'number' && Number.isFinite(value) && value >= -100 && value <= 100) {
          inflationMap[year] = value
        }
      })
    }
  } catch {
    // A fallback is intentionally visible in the resulting provenance metadata.
  }

  const records: RegionRecord[] = []
  const start = new Date('2021-01-01T00:00:00')
  for (let offset = 0; offset < 60; offset += 1) {
    const date = new Date(start)
    date.setMonth(start.getMonth() + offset)
    const year = date.getFullYear()
    const month = date.getMonth() + 1
    const yearMonth = `${year}-${String(month).padStart(2, '0')}`
    const monthData = monthly[yearMonth]
    const averageTemperature = monthData?.temperatures.length
      ? monthData.temperatures.reduce((sum, value) => sum + value, 0) / monthData.temperatures.length
      : 20 - 12 * Math.abs(lat) / 45 + 12 * Math.sin(2 * Math.PI * (month - 4) / 12)
    const precipitation = monthData?.precipitation.length
      ? monthData.precipitation.reduce((sum, value) => sum + value, 0)
      : 55 + 25 * Math.sin(2 * Math.PI * (month - 3) / 12)
    const temperatureProvenance = monthData?.temperatures.length ? 'observed' : 'estimated fallback'
    const precipitationProvenance = monthData?.precipitation.length ? 'observed' : 'estimated fallback'
    const inflation = inflationMap[year] ?? 3
    const inflationProvenance = inflationMap[year] === undefined ? 'estimated fallback' : 'observed'
    const geopoliticalRisk = 0.15
    const newsSentiment = 0.2
    const yieldFactor = 0.9
    let cropYield = 100 * yieldFactor
    if (averageTemperature > 28 && precipitation < 25) cropYield *= 0.65
    cropYield = clamp(cropYield, 0, 140)
    const fuelPrice = 65
    const transportCost = 1 + 0.8 * (fuelPrice - 50) / 50 + 0.5 * geopoliticalRisk
    let price = 100 + (100 - cropYield) * 1.2 + (transportCost - 1) * 45 + inflation * 1.5 + geopoliticalRisk * 30 - newsSentiment * 15
    price = clamp(price, 50, 300)
    const droughtIndex = clamp((averageTemperature - 15) / 5 + (100 - precipitation) / 20, 0, 10)
    const shortageRisk = clamp((100 - cropYield) * 0.4 + (transportCost - 1) * 20 + geopoliticalRisk * 25 - (10 - droughtIndex) * 2, 5, 95)

    records.push({
      year_month: yearMonth,
      year,
      month,
      crop_yield: cropYield,
      fuel_price: fuelPrice,
      transport_cost: transportCost,
      inflation_rate: inflation,
      geopolitical_risk: geopoliticalRisk,
      news_sentiment: newsSentiment,
      food_price_index: price,
      shortage_risk_score: shortageRisk,
      temperature: averageTemperature,
      precipitation,
      region: match.name,
      data_provenance: {
        temperature: temperatureProvenance,
        precipitation: precipitationProvenance,
        inflation_rate: inflationProvenance,
        crop_yield: 'synthetic',
        fuel_price: 'synthetic',
        transport_cost: 'estimated',
        food_price_index: 'synthetic',
        shortage_risk_score: 'synthetic',
      },
    })
  }
  return { region: match.name, records }
}
