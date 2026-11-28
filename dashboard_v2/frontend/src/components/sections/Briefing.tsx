import { useState } from 'react'
import { CloudSun } from 'lucide-react'
import { formatNumber, regionLabel, riskBandName } from '../../lib/format'
import type { Forecast } from '../../types'
import { PromptBarView, TextTypeView } from '../reactBits'

type Props = { region: string; forecast?: Forecast; price: number; risk: number }

/** A template summary of the current figures; no language model is involved. */
export function Briefing({ region, forecast, price, risk }: Props) {
  const [briefing, setBriefing] = useState('Select a region to produce an evidence-led briefing.')
  const [busy, setBusy] = useState(false)

  function submit(message: string) {
    setBusy(true)
    const interval = forecast?.price_interval
      ? ` Price range ${formatNumber(forecast.price_interval.lower)}–${formatNumber(forecast.price_interval.upper)}.`
      : ''
    setBriefing(`${regionLabel(region)}: ${riskBandName(risk)} shortage-risk outlook at ${formatNumber(risk)}%. Three-month price signal ${formatNumber(price)}.${interval} Brief request: ${message || 'Current risk summary'}`)
    window.setTimeout(() => setBusy(false), 260)
  }

  return (
    <div className="briefing">
      <div>
        <CloudSun size={22} />
        <strong>Analyst briefing</strong>
      </div>
      <PromptBarView
        placeholder="Ask for an intervention brief..."
        busy={busy}
        width={480}
        sources={[]}
        commands={[]}
        models={[]}
        efforts={[]}
        onSend={(value: string) => void submit(value)}
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
  )
}
