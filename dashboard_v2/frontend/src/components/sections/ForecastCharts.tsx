import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { ArrowDownRight, ArrowUpRight } from 'lucide-react'
import { formatNumber } from '../../lib/format'
import type { Forecast, RegionRecord } from '../../types'
import { axisTick, gridStroke } from './chartTheme'

type Props = { records: RegionRecord[]; forecasts: Forecast[]; activePrice: number; activeRisk: number }

export function ForecastCharts({ records, forecasts, activePrice, activeRisk }: Props) {
  const history = records.slice(-12).map((record) => ({
    month: record.year_month,
    actualPrice: record.food_price_index,
    actualRisk: record.shortage_risk_score,
  }))
  const projected = forecasts.map((item) => ({
    month: item.target_month,
    forecastPrice: item.predicted_price,
    forecastRisk: item.predicted_shortage_risk,
  }))
  const chartData = [...history, ...projected]
    .filter((item, index, all) => all.findIndex((candidate) => candidate.month === item.month) === index)

  return (
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
              <CartesianGrid strokeDasharray="2 7" stroke={gridStroke} />
              <XAxis dataKey="month" tick={axisTick} tickLine={false} axisLine={false} />
              <YAxis tick={axisTick} tickLine={false} axisLine={false} />
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
              <CartesianGrid strokeDasharray="2 7" stroke={gridStroke} />
              <XAxis dataKey="month" tick={axisTick} tickLine={false} axisLine={false} />
              <YAxis domain={[0, 100]} tick={axisTick} tickLine={false} axisLine={false} />
              <Tooltip />
              <Area type="monotone" dataKey="actualRisk" stroke="#2ba0ff" fill="#2ba0ff" fillOpacity={0.18} strokeWidth={2} />
              <Area type="monotone" dataKey="forecastRisk" stroke="#ff705d" fill="#ff705d" fillOpacity={0.12} strokeWidth={2} strokeDasharray="5 5" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </article>
    </div>
  )
}
