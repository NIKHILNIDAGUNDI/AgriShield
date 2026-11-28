import type { CSSProperties } from 'react'
import { ChevronRight, Database, Gauge, ShieldCheck } from 'lucide-react'
import { outlookMonth } from '../../lib/forecast'
import { formatNumber, regionLabel } from '../../lib/format'
import type { View } from '../../lib/views'
import type { Forecast, RegionRecord } from '../../types'
import { RotatingTextView, ScrambledTextView, TextTypeView } from '../reactBits'
import { RiskBand } from './RiskBand'

const regionColors: Record<string, { color: string; accent: string }> = {
  US: { color: '#69c74f', accent: '#2ba0ff' },
  Ukraine: { color: '#ff705d', accent: '#f5e211' },
  India: { color: '#f5e211', accent: '#ebc1ff' },
}

type Props = {
  regions: string[]
  records: RegionRecord[]
  forecasts: Forecast[]
  activeRegion: string
  activePrice: number
  activeRisk: number
  onSelect: (region: string) => void
  onNavigate: (view: View) => void
}

export function AtlasHero({ regions, records, forecasts, activeRegion, activePrice, activeRisk, onSelect, onNavigate }: Props) {
  return (
    <section id="overview" className="atlas-viewport">
      <div className="atlas-copy">
        <div className="atlas-kicker">
          <Database size={16} />
          Three-month outlook
        </div>
        <h1>
          <TextTypeView
            text={[`${regionLabel(activeRegion)} is`, 'Risk, made visible.', 'Forecasting food security.']}
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
          <button className="primary-action" onClick={() => onNavigate('analytics')}>
            Inspect the evidence <ChevronRight size={18} />
          </button>
          <button className="secondary-action" onClick={() => onNavigate('simulator')}>
            Run scenario
          </button>
        </div>
      </div>

      <div className="atlas-map" aria-label="Regional forecast map">
        {regions.slice(0, 5).map((region, index) => {
          const forecast = forecasts.find((item) => item.region === region && item.target_month === outlookMonth)
          const record = records.find((item) => item.region === region && item.year_month === '2025-12')
          const risk = forecast?.predicted_shortage_risk ?? record?.shortage_risk_score ?? 5
          const palette = regionColors[region] ?? { color: index % 2 ? '#2ba0ff' : '#ebc1ff', accent: '#ff705d' }
          return (
            <button
              key={region}
              className={`region-lobe region-lobe-${index % 4} ${activeRegion === region ? 'active' : ''}`}
              style={{ '--lobe': palette.color, '--lobe-accent': palette.accent } as CSSProperties}
              onClick={() => onSelect(region)}
              aria-pressed={activeRegion === region}
            >
              <ScrambledTextView radius={90} duration={0.8} className="lobe-name">
                {regionLabel(region)}
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
  )
}
