import { AlertTriangle } from 'lucide-react'
import { heatmapMonths } from '../../lib/forecast'
import { formatNumber, regionLabel } from '../../lib/format'
import type { Forecast, RegionRecord } from '../../types'

type Props = { regions: string[]; records: RegionRecord[]; forecasts: Forecast[]; activeRisk: number }

export function RiskMap({ regions, records, forecasts, activeRisk }: Props) {
  const heatmap = regions.map((region) => ({
    region,
    values: heatmapMonths.map((month) => (
      month.startsWith('2026')
        ? forecasts.find((item) => item.region === region && item.target_month === month)?.predicted_shortage_risk
        : records.find((item) => item.region === region && item.year_month === month)?.shortage_risk_score
    ) ?? 5),
  }))

  return (
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
                  <th>{regionLabel(row.region)}</th>
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
  )
}
