import { useState } from 'react'
import { predictFeatures } from '../lib/api'
import { buildScenario, defaultControls, type ScenarioControls } from '../lib/simulator'
import type { RegionRecord } from '../types'

type Result = { key: string; price: number; risk: number; source: 'model' | 'estimate' }

export function useSimulator(
  region: string,
  records: RegionRecord[],
  apiAvailable: boolean,
  onApiFailure: () => void,
) {
  const [edited, setEdited] = useState<{ region: string; controls: ScenarioControls } | null>(null)
  const [stored, setStored] = useState<Result | null>(null)

  const latest = records.at(-1)
  const controls = edited?.region === region ? edited.controls : latest ? defaultControls(latest) : null
  const key = JSON.stringify([region, controls])
  // A result is shown only while it still matches the region and sliders on screen.
  const result = stored?.key === key ? stored : null

  function setControl(name: keyof ScenarioControls, value: number) {
    if (controls) setEdited({ region, controls: { ...controls, [name]: value } })
  }

  async function run() {
    if (!controls || records.length < 3) return
    const scenario = buildScenario(records, controls)
    if (apiAvailable) {
      try {
        const [prediction] = (await predictFeatures(region, scenario.features)).predictions
        setStored({
          key,
          price: prediction.predicted_price,
          risk: prediction.predicted_shortage_risk,
          source: 'model',
        })
        return
      } catch {
        onApiFailure()
      }
    }
    setStored({ key, ...scenario.estimate, source: 'estimate' })
  }

  return { controls, setControl, result, run, baselineMonth: latest?.year_month }
}
