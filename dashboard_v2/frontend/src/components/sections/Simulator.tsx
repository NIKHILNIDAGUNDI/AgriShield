import { ChevronRight, Gauge } from 'lucide-react'
import { useSimulator } from '../../hooks/useSimulator'
import { outlookMonth } from '../../lib/forecast'
import { formatNumber, monthLabel, regionLabel } from '../../lib/format'
import { scenarioTargetMonth } from '../../lib/simulator'
import type { RegionRecord } from '../../types'
import { RiskBand } from './RiskBand'

const sliders = [
  { name: 'temperatureShift', label: 'Temperature shift', min: -15, max: 15, step: 0.5, digits: 1, suffix: '°C' },
  { name: 'precipitationShift', label: 'Precipitation shift', min: -100, max: 100, step: 1, digits: 0, suffix: '%' },
  { name: 'inflation', label: 'Inflation', min: -10, max: 30, step: 0.1, digits: 1, suffix: '%' },
  { name: 'fuel', label: 'Fuel price', min: 20, max: 150, step: 1, digits: 0, suffix: '$' },
  { name: 'geopolitical', label: 'Geopolitical risk', min: 0, max: 1, step: 0.05, digits: 2, suffix: '' },
  { name: 'sentiment', label: 'News sentiment', min: -1, max: 1, step: 0.05, digits: 2, suffix: '' },
] as const

type Props = {
  region: string
  records: RegionRecord[]
  apiAvailable: boolean
  onApiFailure: () => void
}

export function Simulator({ region, records, apiAvailable, onApiFailure }: Props) {
  const { controls, setControl, result, run, baselineMonth } = useSimulator(region, records, apiAvailable, onApiFailure)

  const targetMonth = baselineMonth ? monthLabel(scenarioTargetMonth(baselineMonth)) : ''

  return (
    <div className="simulator-grid">
      <div className="simulator-controls">
        {controls && sliders.map(({ name, label, min, max, step, digits, suffix }) => (
          <label className="slider-row" key={name}>
            <span>{label}</span>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={controls![name]}
              onChange={(event) => setControl(name, Number(event.target.value))}
            />
            <output>{controls![name].toFixed(digits)}{suffix}</output>
          </label>
        ))}
        {baselineMonth && (
          <div className="simulator-baseline">
            <p>Changes apply to the {monthLabel(baselineMonth)} readings for {regionLabel(region)}.</p>
            {apiAvailable ? (
              <p>
                <strong>Forecast month: {targetMonth}.</strong> This is one month after the{' '}
                {monthLabel(outlookMonth)} outlook card, so the figures can differ even with unchanged sliders.
              </p>
            ) : (
              <p>API unavailable: the result is a rough in-browser estimate, not a forecast for a specific month.</p>
            )}
          </div>
        )}
        <button className="primary-action" onClick={() => void run()}>
          Recalculate <ChevronRight size={18} />
        </button>
      </div>

      <div className="simulation-result" aria-live="polite">
        {result ? (
          <>
          <div className="result-orbit">
            <span>{result.source === 'model' ? `${targetMonth} price` : 'Estimated price'}</span>
            <strong>{formatNumber(result.price)}</strong>
          </div>
          <div className="result-readout">
            <Gauge size={24} />
            <span>Shortage risk</span>
            <strong>{formatNumber(result.risk)}%</strong>
            <RiskBand risk={result.risk} />
            <small>{result.source === 'model' ? `Model prediction for ${targetMonth}` : 'In-browser estimate (API unavailable)'}</small>
          </div>
          </>
        ) : (
          <p className="simulation-empty">Set the conditions, then press Recalculate to see a projection.</p>
        )}
      </div>
    </div>
  )
}
