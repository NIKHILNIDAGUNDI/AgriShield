import { Activity } from 'lucide-react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { TrainingMetrics } from '../../types'
import { axisTick, gridStroke } from './chartTheme'

const models = [
  { key: 'rf', label: 'Random Forest' },
  { key: 'torch', label: 'PyTorch MLP' },
] as const

export function ModelComparison({ results }: { results: TrainingMetrics | null }) {
  const folds = results?.validation?.n_splits
  const rows = results
    ? models.map(({ key, label }) => ({
        model: label,
        price: results.metrics[`${key}_price_rmse`],
        risk: results.metrics[`${key}_risk_rmse`],
        priceR2: results.metrics[`${key}_price_r2`],
        riskR2: results.metrics[`${key}_risk_r2`],
      }))
    : []

  return (
    <article className="model-comparison">
      <div className="panel-title">
        <div>
          <span>Model comparison</span>
          <strong>{folds ? `Walk-forward validation, mean of ${folds} folds` : 'Walk-forward validation'}</strong>
        </div>
        <Activity size={20} />
      </div>
      {results ? (
        <>
          <div className="comparison-bars">
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={rows} layout="vertical">
                <CartesianGrid strokeDasharray="2 7" stroke={gridStroke} />
                <XAxis type="number" tick={axisTick} />
                <YAxis type="category" dataKey="model" tick={axisTick} width={80} />
                <Tooltip formatter={(value) => Number(value).toFixed(2)} />
                <Legend formatter={(value) => <span className="legend-label">{value}</span>} />
                <Bar dataKey="price" name="Price RMSE" fill="#69c74f" radius={[0, 8, 8, 0]} />
                <Bar dataKey="risk" name="Risk RMSE" fill="#ff705d" radius={[0, 8, 8, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <table className="metrics-table">
            <thead>
              <tr><th>Model</th><th>Price RMSE</th><th>Price R²</th><th>Risk RMSE</th><th>Risk R²</th></tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.model}>
                  <th>{row.model}</th>
                  <td>{row.price.toFixed(2)}</td>
                  <td>{row.priceR2.toFixed(3)}</td>
                  <td>{row.risk.toFixed(2)}</td>
                  <td>{row.riskR2.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <p className="metrics-empty">Validation metrics are unavailable.</p>
      )}
    </article>
  )
}
