import { riskBandName } from '../../lib/format'

export function RiskBand({ risk }: { risk: number }) {
  const band = riskBandName(risk)
  return (
    <span className={`risk-band risk-${band.toLowerCase()}`}>
      <span className="risk-dot" />
      {band}
    </span>
  )
}
