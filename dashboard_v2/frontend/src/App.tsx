import { useEffect, useMemo, useState } from 'react'
import type { ComponentType } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  ChevronRight,
  CloudSun,
  Database,
  Gauge,
  Globe2,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Wheat,
} from 'lucide-react'

import DotField from './components/DotField.jsx'
import GlowCursor from './components/GlowCursor.jsx'
import GooeyNav from './components/GooeyNav.jsx'
import LatticeLoader from './components/LatticeLoader.jsx'
import PromptBar from './components/PromptBar.jsx'
import RotatingText from './components/RotatingText.jsx'
import ScrambledText from './components/ScrambledText.jsx'
import TextType from './components/TextType.jsx'
import { fetchDataset, fetchHealth, fetchModels, predictFeatures, predictRows } from './lib/api'
import { createCustomRegion } from './lib/customRegion'
import type { ApiModels, Forecast, RegionRecord } from './types'

const DotFieldView = DotField as unknown as ComponentType<Record<string, unknown>>
const GlowCursorView = GlowCursor as unknown as ComponentType<Record<string, unknown>>
const GooeyNavView = GooeyNav as unknown as ComponentType<Record<string, unknown>>
const LatticeLoaderView = LatticeLoader as unknown as ComponentType<Record<string, unknown>>
const PromptBarView = PromptBar as unknown as ComponentType<Record<string, unknown>>
const RotatingTextView = RotatingText as unknown as ComponentType<Record<string, unknown>>
const ScrambledTextView = ScrambledText as unknown as ComponentType<Record<string, unknown>>
const TextTypeView = TextType as unknown as ComponentType<Record<string, unknown>>

type View = 'overview' | 'analytics' | 'simulator'

const sourceFeatures = [
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

const regionColors: Record<string, { color: string; accent: string }> = {
  US: { color: '#69c74f', accent: '#2ba0ff' },
  Ukraine: { color: '#ff705d', accent: '#f5e211' },
  India: { color: '#f5e211', accent: '#ebc1ff' },
}

const targetMonths = ['2026-01', '2026-02', '2026-03']
const heatmapMonths = [
  '2025-01', '2025-02', '2025-03', '2025-04', '2025-05', '2025-06',
  '2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12',
  ...targetMonths,
]

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value))
}

function regionSortName(region: string) {
  if (region === 'US') return 'United States'
  return region
}

function staticFallback(records: RegionRecord[]) {
  const regions = [...new Set(records.map((record) => record.region))]
  return regions.flatMap((region) =>
    targetMonths.map((targetMonth, index) => {
      const regionRows = records.filter((record) => record.region === region).sort((a, b) => a.year_month.localeCompare(b.year_month))
      const recent = regionRows.slice(-3)
      const priceValues = recent.map((row) => row.food_price_index)
      const riskValues = recent.map((row) => row.shortage_risk_score)
      const priceTrend = recent.length > 1 ? (priceValues.at(-1)! - priceValues[0]) / 2 : 0
      const riskTrend = recent.length > 1 ? (riskValues.at(-1)! - riskValues[0]) / 2 : 0
      return {
        region,
        feature_month: recent.at(-1)?.year_month ?? '2025-12',
        target_month: targetMonth,
        predicted_price: clamp(priceValues.at(-1)! + priceTrend * (index + 1), 50, 300),
        predicted_shortage_risk: clamp(riskValues.at(-1)! + riskTrend * (index + 1), 0, 100),
      }
    }),
  )
}

function formatNumber(value: number | null | undefined, digits = 1) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

function RiskBand({ risk }: { risk: number }) {
  const band = risk >= 65 ? 'Critical' : risk >= 30 ? 'Elevated' : 'Secure'
  return (
    <span className={`risk-band risk-${band.toLowerCase()}`}>
      <span className="risk-dot" />
      {band}
    </span>
  )
}

function App() {
  const [records, setRecords] = useState<RegionRecord[]>([])
  const [forecasts, setForecasts] = useState<Forecast[]>([])
  const [health, setHealth] = useState<{ status: string; active_model_version: string } | null>(null)
  const [models, setModels] = useState<ApiModels | null>(null)
  const [activeRegion, setActiveRegion] = useState('US')
  const [activeView, setActiveView] = useState<View>('overview')
  const [loading, setLoading] = useState(true)
  const [apiAvailable, setApiAvailable] = useState(false)
  const [customName, setCustomName] = useState('')
  const [customStatus, setCustomStatus] = useState('')
  const [briefing, setBriefing] = useState('Select a region to produce an evidence-led briefing.')
  const [briefingBusy, setBriefingBusy] = useState(false)
  const [simulator, setSimulator] = useState({ temperature: 0, precipitation: 0, inflation: 2.5, fuel: 65, geo: 0.05, sentiment: 0.4 })
  const [simulation, setSimulation] = useState({ price: 0, risk: 0 })

  useEffect(() => {
    let cancelled = false
    async function initialize() {
      try {
        const dataset = await fetchDataset()
        if (cancelled) return
        setRecords(dataset)
        setLoading(false)

        const [healthResult, modelsResult] = await Promise.allSettled([fetchHealth(), fetchModels()])
        if (cancelled) return
        if (healthResult.status === 'fulfilled') setHealth(healthResult.value)
        if (modelsResult.status === 'fulfilled') setModels(modelsResult.value)

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
        setCustomStatus(error instanceof Error ? error.message : 'Dashboard data could not be loaded.')
      }
    }
    void initialize()
    return () => { cancelled = true }
  }, [])

  const regions = useMemo(
    () => [...new Set(records.map((record) => record.region))],
    [records],
  )

  const activeRecords = useMemo(
    () => records.filter((record) => record.region === activeRegion).sort((a, b) => a.year_month.localeCompare(b.year_month)),
    [activeRegion, records],
  )

  const activeForecasts = useMemo(
    () => forecasts.filter((forecast) => forecast.region === activeRegion).sort((a, b) => a.target_month.localeCompare(b.target_month)),
    [activeRegion, forecasts],
  )

  const latestRecord = activeRecords.at(-1)
  const marchForecast = activeForecasts.find((forecast) => forecast.target_month === '2026-03')
  const activePrice = marchForecast?.predicted_price ?? latestRecord?.food_price_index ?? 0
  const activeRisk = marchForecast?.predicted_shortage_risk ?? latestRecord?.shortage_risk_score ?? 0

  const chartData = useMemo(() => {
    const history = activeRecords.slice(-12).map((record) => ({
      month: record.year_month,
      actualPrice: record.food_price_index,
      actualRisk: record.shortage_risk_score,
    }))
    const forecast = activeForecasts.map((item) => ({
      month: item.target_month,
      forecastPrice: item.predicted_price,
      forecastRisk: item.predicted_shortage_risk,
    }))
    return [...history, ...forecast].filter((item, index, array) => array.findIndex((candidate) => candidate.month === item.month) === index)
  }, [activeForecasts, activeRecords])

  const heatmap = useMemo(() => regions.map((region) => {
    const regionRecords = records.filter((record) => record.region === region)
    const regionForecasts = forecasts.filter((forecast) => forecast.region === region)
    const values = heatmapMonths.map((month) => {
      if (month.startsWith('2026')) {
        return regionForecasts.find((forecast) => forecast.target_month === month)?.predicted_shortage_risk ?? 5
      }
      return regionRecords.find((record) => record.year_month === month)?.shortage_risk_score ?? 5
    })
    return { region, values }
  }), [forecasts, records, regions])

  async function handleCustomRegion() {
    if (!customName.trim()) {
      setCustomStatus('Enter a region name to begin.')
      return
    }
    setCustomStatus('Resolving region and climatology...')
    try {
      const result = await createCustomRegion(customName)
      setCustomStatus(`Fetched ${result.region}. Calibrating forecast...`)
      setRecords((current) => [...current, ...result.records])
      try {
        const prediction = await predictRows(result.records, true)
        setForecasts((current) => [
          ...current.filter((forecast) => forecast.region !== result.region),
          ...prediction.predictions,
        ])
        setApiAvailable(true)
      } catch {
        setApiAvailable(false)
        setForecasts((current) => [
          ...current.filter((forecast) => forecast.region !== result.region),
          ...staticFallback(result.records),
        ])
      }
      setActiveRegion(result.region)
      setCustomStatus(`Active: ${result.region}. Observed weather and CPI are labeled in region provenance.`)
      setCustomName('')
    } catch (error) {
      setCustomStatus(error instanceof Error ? error.message : 'Custom-region lookup failed.')
    }
  }

  async function runBriefing(message: string) {
    setBriefingBusy(true)
    const regionName = regionSortName(activeRegion)
    const interval = marchForecast?.price_interval
      ? ` Price range ${formatNumber(marchForecast.price_interval.lower)}–${formatNumber(marchForecast.price_interval.upper)}.`
      : ''
    const summary = `${regionName}: ${RiskBandName(activeRisk)} shortage-risk outlook at ${formatNumber(activeRisk)}%. Three-month price signal ${formatNumber(activePrice)}.${interval} Brief request: ${message || 'Current risk summary'}`
    setBriefing(summary)
    window.setTimeout(() => setBriefingBusy(false), 260)
  }

  async function runSimulation() {
    const defaults = activeRegion === 'US'
      ? { cropYield: 100, geoRisk: 0.05 }
      : activeRegion === 'Ukraine'
        ? { cropYield: 75, geoRisk: 0.95 }
        : { cropYield: 90, geoRisk: 0.2 }
    const drought = simulator.precipitation < -40 || simulator.temperature > 2.5
    const cropYield = drought ? defaults.cropYield * 0.65 : defaults.cropYield
    const transportCost = 1 + 0.8 * (simulator.fuel - 50) / 50 + 0.5 * simulator.geo
    const price = clamp(100 + (100 - cropYield) * 1.2 + (transportCost - 1) * 45 + simulator.inflation * 1.5 + simulator.geo * 30 - simulator.sentiment * 15, 50, 300)
    const droughtIndex = clamp((simulator.temperature + 15) / 5 + (100 - (simulator.precipitation + 100) / 2) / 20, 0, 10)
    const risk = clamp((100 - cropYield) * 0.4 + (transportCost - 1) * 20 + simulator.geo * 25 - (10 - droughtIndex) * 2, 5, 95)

    if (apiAvailable && latestRecord) {
      const simulatedState: Record<string, number> = {
        temperature: simulator.temperature,
        precipitation: (simulator.precipitation + 100) / 2 * 10,
        crop_yield: cropYield,
        fuel_price: simulator.fuel,
        transport_cost: transportCost,
        inflation_rate: simulator.inflation,
        geopolitical_risk: simulator.geo,
        news_sentiment: simulator.sentiment,
        food_price_index: price,
        shortage_risk_score: risk,
      }
      const features: Record<string, number> = {}
      sourceFeatures.forEach((source) => {
        features[`${source}_lag_1`] = source === 'temperature'
          ? simulatedState[source]
          : source === 'precipitation'
            ? simulatedState[source]
            : source === 'crop_yield'
              ? cropYield
              : source === 'fuel_price'
                ? simulator.fuel
                : source === 'transport_cost'
                  ? transportCost
                  : source === 'inflation_rate'
                    ? simulator.inflation
                    : source === 'geopolitical_risk'
                      ? simulator.geo
                      : source === 'news_sentiment'
                        ? simulator.sentiment
                        : source === 'food_price_index'
                          ? price
                          : risk
        features[`${source}_lag_2`] = latestRecord[`${source}_lag_1` as keyof RegionRecord] as number || latestRecord[source]
        features[`${source}_lag_3`] = latestRecord[`${source}_lag_2` as keyof RegionRecord] as number || latestRecord[source]
      })
      try {
        const response = await predictFeatures(activeRegion, features)
        const prediction = response.predictions[0]
        setSimulation({ price: prediction.predicted_price, risk: prediction.predicted_shortage_risk })
        return
      } catch {
        setApiAvailable(false)
      }
    }
    setSimulation({ price, risk })
  }

  const navItems = [
    { label: 'Overview', href: '#overview' },
    { label: 'Analytics', href: '#analytics' },
    { label: 'Simulator', href: '#simulator' },
  ]

  return (
    <GlowCursorView
      theme="light"
      color="#69c74f"
      secondaryColor="#ff705d"
      intensity={0.35}
      trailWidth={2}
      className="cursor-shell"
    >
      <div className="app-shell">
        <DotFieldView
          className="dot-field"
          pattern="scatter"
          dotRadius={0.9}
          dotSpacing={18}
          gradientFrom="#69c74f"
          gradientTo="#ff705d"
          baseOpacity={0.16}
          lightMode
          interactive
        />

        <header className="topbar">
          <div className="brand">
            <span className="brand-mark"><Wheat size={18} /></span>
            <span>AgriShield</span>
          </div>
          <GooeyNavView
            items={navItems}
            theme="light"
            color="#69c74f"
            activeTextColor="#1f2a25"
            activeIndex={activeView === 'overview' ? 0 : activeView === 'analytics' ? 1 : 2}
            onChange={(index: number) => {
              const view: View = index === 0 ? 'overview' : index === 1 ? 'analytics' : 'simulator'
              setActiveView(view)
              document.querySelector(`#${view}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
            }}
          />
          <div className={`api-state ${apiAvailable ? 'online' : 'fallback'}`} role="status" aria-live="polite">
            <span />
            {apiAvailable
              ? `API · ${models?.active_model_version ?? health?.active_model_version ?? 'connected'}`
              : 'Static fallback'}
          </div>
        </header>

        <main>
          {loading ? (
            <section className="loading-state" aria-live="polite">
              <LatticeLoaderView label="Calibrating" status="working" pattern="ripple" grid={3} color="#69c74f" />
            </section>
          ) : (
            <>
              <section id="overview" className="atlas-viewport">
                <div className="atlas-copy">
                  <div className="atlas-kicker">
                    <Database size={16} />
                    Three-month outlook
                  </div>
                  <h1>
                    <TextTypeView
                      text={[`${regionSortName(activeRegion)} is`, 'Risk, made visible.', 'Forecasting food security.']}
                      typingSpeed={38}
                      pauseDuration={1800}
                      deletingSpeed={28}
                      showCursor
                      cursorCharacter=" "
                    />
                  </h1>
                  <RotatingTextView
                    texts={['food price pressure', 'regional shortage risk', 'climate and market stress']}
                    mainClassName="rotating-lead"
                    effect="blur"
                    rotationInterval={2200}
                  />
                  <div className="atlas-actions">
                    <button
                      className="primary-action"
                      onClick={() => {
                        setActiveView('analytics')
                        document.querySelector('#analytics')?.scrollIntoView({ behavior: 'smooth' })
                      }}
                    >
                      Inspect the evidence <ChevronRight size={18} />
                    </button>
                    <button
                      className="secondary-action"
                      onClick={() => {
                        setActiveView('simulator')
                        document.querySelector('#simulator')?.scrollIntoView({ behavior: 'smooth' })
                      }}
                    >
                      Run scenario
                    </button>
                  </div>
                </div>

                <div className="atlas-map" aria-label="Regional forecast map">
                  {regions.slice(0, 5).map((region, index) => {
                    const forecast = forecasts.find((item) => item.region === region && item.target_month === '2026-03')
                    const record = records.find((item) => item.region === region && item.year_month === '2025-12')
                    const risk = forecast?.predicted_shortage_risk ?? record?.shortage_risk_score ?? 5
                    const palette = regionColors[region] ?? { color: index % 2 ? '#2ba0ff' : '#ebc1ff', accent: '#ff705d' }
                    return (
                      <button
                        key={region}
                        className={`region-lobe region-lobe-${index % 4} ${activeRegion === region ? 'active' : ''}`}
                        style={{ '--lobe': palette.color, '--lobe-accent': palette.accent } as React.CSSProperties}
                        onClick={() => setActiveRegion(region)}
                        aria-pressed={activeRegion === region}
                      >
                        <ScrambledTextView radius={90} duration={0.8} className="lobe-name">
                          {regionSortName(region)}
                        </ScrambledTextView>
                        <strong>{formatNumber(risk)}%</strong>
                        <RiskBand risk={risk} />
                      </button>
                    )
                  })}
                </div>

                <aside className="atlas-readout">
                  <div className="readout-head">
                    <span>March 2026</span>
                    <ShieldCheck size={20} />
                  </div>
                  <div className="readout-value">
                    <strong>{formatNumber(activePrice)}</strong>
                    <span>price index</span>
                  </div>
                  <div className="readout-risk">
                    <Gauge size={18} />
                    <div>
                      <strong>{formatNumber(activeRisk)}%</strong>
                      <span>shortage risk</span>
                    </div>
                  </div>
                  <RiskBand risk={activeRisk} />
                </aside>
              </section>

              <section className="region-strip" aria-label="Region selector">
                {regions.map((region) => (
                  <button key={region} className={activeRegion === region ? 'active' : ''} onClick={() => setActiveRegion(region)}>
                    <Globe2 size={16} />
                    {regionSortName(region)}
                  </button>
                ))}
              </section>

              <section className="custom-region" aria-labelledby="custom-region-heading">
                <div>
                  <div className="section-index">Location brief</div>
                  <h2 id="custom-region-heading">Analyze another place</h2>
                </div>
                <div className="custom-form">
                  <label htmlFor="custom-region-input">Region name</label>
                  <div className="search-field">
                    <Search size={18} />
                    <input
                      id="custom-region-input"
                      value={customName}
                      onChange={(event) => setCustomName(event.target.value)}
                      onKeyDown={(event) => { if (event.key === 'Enter') void handleCustomRegion() }}
                      placeholder="Spain, Kenya, Tokyo..."
                    />
                    <button onClick={() => void handleCustomRegion()} aria-label="Fetch custom region forecast">
                      <ChevronRight size={18} />
                    </button>
                  </div>
                  <p className="custom-status" role="status">{customStatus || 'Observed weather and CPI are retained as observed; generated fields are labeled.'}</p>
                </div>
              </section>

              <section id="analytics" className="analytics-section">
                <div className="section-heading">
                  <div>
                    <div className="section-index">Evidence</div>
                    <h2>{regionSortName(activeRegion)} forecast trail</h2>
                  </div>
                  <RiskBand risk={activeRisk} />
                </div>

                <div className="analytics-grid">
                  <article className="chart-panel price-panel">
                    <div className="panel-title">
                      <div>
                        <span>Price index</span>
                        <strong>{formatNumber(activePrice)}</strong>
                      </div>
                      <span className="trend-up"><ArrowUpRight size={15} /> 3-month signal</span>
                    </div>
                    <div className="chart-wrap">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
                          <CartesianGrid strokeDasharray="2 7" stroke="rgba(31,42,37,.12)" />
                          <XAxis dataKey="month" tick={{ fill: '#667267', fontSize: 11 }} tickLine={false} axisLine={false} />
                          <YAxis tick={{ fill: '#667267', fontSize: 11 }} tickLine={false} axisLine={false} />
                          <Tooltip />
                          <Area type="monotone" dataKey="actualPrice" stroke="#1f2a25" fill="#69c74f" fillOpacity={0.22} strokeWidth={2} />
                          <Area type="monotone" dataKey="forecastPrice" stroke="#ff705d" fill="#ff705d" fillOpacity={0.12} strokeWidth={2} strokeDasharray="5 5" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </article>

                  <article className="chart-panel risk-panel">
                    <div className="panel-title">
                      <div>
                        <span>Shortage risk</span>
                        <strong>{formatNumber(activeRisk)}%</strong>
                      </div>
                      <span className="trend-down"><ArrowDownRight size={15} /> monitored</span>
                    </div>
                    <div className="chart-wrap">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={chartData}>
                          <CartesianGrid strokeDasharray="2 7" stroke="rgba(31,42,37,.12)" />
                          <XAxis dataKey="month" tick={{ fill: '#667267', fontSize: 11 }} tickLine={false} axisLine={false} />
                          <YAxis domain={[0, 100]} tick={{ fill: '#667267', fontSize: 11 }} tickLine={false} axisLine={false} />
                          <Tooltip />
                          <Area type="monotone" dataKey="actualRisk" stroke="#2ba0ff" fill="#2ba0ff" fillOpacity={0.18} strokeWidth={2} />
                          <Area type="monotone" dataKey="forecastRisk" stroke="#ff705d" fill="#ff705d" fillOpacity={0.12} strokeWidth={2} strokeDasharray="5 5" />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>
                  </article>
                </div>

                <div className="heatmap-and-alerts">
                  <article className="heatmap-panel">
                    <div className="panel-title">
                      <div>
                        <span>Regional risk map</span>
                        <strong>Jan 2025 — Mar 2026</strong>
                      </div>
                    </div>
                    <div className="heatmap-scroll">
                      <table>
                        <thead>
                          <tr>
                            <th>Region</th>
                            {heatmapMonths.map((month) => <th key={month}>{month.slice(5)}'{month.slice(2, 4)}</th>)}
                          </tr>
                        </thead>
                        <tbody>
                          {heatmap.map((row) => (
                            <tr key={row.region}>
                              <th>{regionSortName(row.region)}</th>
                              {row.values.map((risk, index) => (
                                <td
                                  key={`${row.region}-${index}`}
                                  className={risk >= 65 ? 'cell-critical' : risk >= 30 ? 'cell-elevated' : 'cell-secure'}
                                  title={`${formatNumber(risk)}% shortage risk`}
                                >
                                  {formatNumber(risk, 0)}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </article>

                  <aside className="alerts-panel">
                    <div className="panel-title">
                      <div>
                        <span>Early warning</span>
                        <strong>{activeRisk >= 30 ? 'Intervention review' : 'Watch list only'}</strong>
                      </div>
                      <AlertTriangle size={20} />
                    </div>
                    <div className="alert-list">
                      {activeRisk >= 65 ? (
                        <div className="alert-row critical"><strong>Critical shortage signal</strong><span>Review supply and intervention capacity.</span></div>
                      ) : activeRisk >= 30 ? (
                        <div className="alert-row elevated"><strong>Elevated pressure</strong><span>Monitor inflation, fuel, and yield drivers.</span></div>
                      ) : (
                        <div className="alert-row secure"><strong>No immediate critical alert</strong><span>Continue monthly monitoring.</span></div>
                      )}
                    </div>
                  </aside>
                </div>

                <article className="model-comparison">
                  <div className="panel-title">
                    <div>
                      <span>Model comparison</span>
                      <strong>Holdout performance</strong>
                    </div>
                    <Activity size={20} />
                  </div>
                  <div className="comparison-bars">
                    <ResponsiveContainer width="100%" height={220}>
                      <BarChart
                        data={[
                          { model: 'Random Forest', price: 19.15, risk: 5.565 },
                          { model: 'PyTorch MLP', price: 18.39, risk: 4.83 },
                        ]}
                        layout="vertical"
                      >
                        <CartesianGrid strokeDasharray="2 7" stroke="rgba(31,42,37,.12)" />
                        <XAxis type="number" tick={{ fill: '#667267', fontSize: 11 }} />
                        <YAxis type="category" dataKey="model" tick={{ fill: '#667267', fontSize: 11 }} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="price" name="Price RMSE" fill="#69c74f" radius={[0, 8, 8, 0]} />
                        <Bar dataKey="risk" name="Risk RMSE" fill="#ff705d" radius={[0, 8, 8, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </article>

                <section className="provenance">
                  <div>
                    <span className="provenance-dot observed" />
                    Observed
                  </div>
                  <div>
                    <span className="provenance-dot estimated" />
                    Estimated fallback
                  </div>
                  <div>
                    <span className="provenance-dot synthetic" />
                    Synthetic
                  </div>
                </section>
              </section>

              <section id="simulator" className="simulator-section">
                <div className="section-heading">
                  <div>
                    <div className="section-index">Scenario lab</div>
                    <h2>Move the conditions</h2>
                  </div>
                  <SlidersHorizontal size={22} />
                </div>
                <div className="simulator-grid">
                  <div className="simulator-controls">
                    {[
                      ['temperature', 'Temperature shift', -15, 15, '°C'],
                      ['precipitation', 'Precipitation shift', -100, 100, '%'],
                      ['inflation', 'Inflation', 0, 30, '%'],
                      ['fuel', 'Fuel price', 20, 150, '$'],
                      ['geo', 'Geopolitical risk', 0, 1, ''],
                      ['sentiment', 'News sentiment', -1, 1, ''],
                    ].map(([key, label, min, max, suffix]) => (
                      <label className="slider-row" key={String(key)}>
                        <span>{String(label)}</span>
                        <input
                          type="range"
                          min={Number(min)}
                          max={Number(max)}
                          step={key === 'geo' || key === 'sentiment' ? 0.05 : 1}
                          value={simulator[key as keyof typeof simulator]}
                          onChange={(event) => setSimulator((current) => ({ ...current, [key]: Number(event.target.value) }))}
                        />
                        <output>{Number(simulator[key as keyof typeof simulator]).toFixed(key === 'geo' || key === 'sentiment' ? 2 : 0)}{String(suffix)}</output>
                      </label>
                    ))}
                    <button className="primary-action" onClick={() => void runSimulation()}>
                      Recalculate <ChevronRight size={18} />
                    </button>
                  </div>

                  <div className="simulation-result">
                    <div className="result-orbit">
                      <span>Projected price</span>
                      <strong>{formatNumber(simulation.price)}</strong>
                    </div>
                    <div className="result-readout">
                      <Gauge size={24} />
                      <span>Shortage risk</span>
                      <strong>{formatNumber(simulation.risk)}%</strong>
                      <RiskBand risk={simulation.risk} />
                    </div>
                  </div>
                </div>

                <div className="briefing">
                  <div>
                    <CloudSun size={22} />
                    <strong>Analyst briefing</strong>
                  </div>
                  <PromptBarView
                    placeholder="Ask for an intervention brief..."
                    busy={briefingBusy}
                    width={480}
                    onSend={(value: string) => void runBriefing(value)}
                  />
                  <TextTypeView
                    text={[briefing]}
                    typingSpeed={12}
                    initialDelay={100}
                    pauseDuration={3000}
                    deletingSpeed={20}
                    loop={false}
                    showCursor
                  />
                </div>
              </section>
            </>
          )}
        </main>
      </div>
    </GlowCursorView>
  )
}

function RiskBandName(risk: number) {
  return risk >= 65 ? 'Critical' : risk >= 30 ? 'Elevated' : 'Secure'
}

export default App
