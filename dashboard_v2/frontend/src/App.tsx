import { useState } from 'react'
import { SlidersHorizontal } from 'lucide-react'

import { DotFieldView, GlowCursorView, LatticeLoaderView } from './components/reactBits'
import { AtlasHero } from './components/sections/AtlasHero'
import { Briefing } from './components/sections/Briefing'
import { CustomRegion } from './components/sections/CustomRegion'
import { ForecastCharts } from './components/sections/ForecastCharts'
import { ModelComparison } from './components/sections/ModelComparison'
import { Provenance } from './components/sections/Provenance'
import { RegionStrip } from './components/sections/RegionStrip'
import { RiskBand } from './components/sections/RiskBand'
import { RiskMap } from './components/sections/RiskMap'
import { Simulator } from './components/sections/Simulator'
import { TopBar } from './components/sections/TopBar'
import { useDashboardData } from './hooks/useDashboardData'
import { outlookMonth, regionRecords } from './lib/forecast'
import { regionLabel } from './lib/format'
import type { View } from './lib/views'

function App() {
  const data = useDashboardData()
  const [activeRegion, setActiveRegion] = useState('US')
  const [activeView, setActiveView] = useState<View>('overview')

  const regions = [...new Set(data.records.map((record) => record.region))]
  const activeRecords = regionRecords(data.records, activeRegion)
  const activeForecasts = data.forecasts
    .filter((forecast) => forecast.region === activeRegion)
    .sort((a, b) => a.target_month.localeCompare(b.target_month))
  const outlook = activeForecasts.find((forecast) => forecast.target_month === outlookMonth)
  const latest = activeRecords.at(-1)
  const activePrice = outlook?.predicted_price ?? latest?.food_price_index ?? 0
  const activeRisk = outlook?.predicted_shortage_risk ?? latest?.shortage_risk_score ?? 0

  function navigate(view: View) {
    setActiveView(view)
    document.querySelector(`#${view}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

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

        <TopBar
          activeView={activeView}
          onNavigate={navigate}
          apiAvailable={data.apiAvailable}
          modelVersion={data.modelVersion}
        />

        <main>
          {data.loading ? (
            <section className="loading-state" aria-live="polite">
              <LatticeLoaderView label="Calibrating" status="working" pattern="ripple" grid={3} color="#69c74f" />
            </section>
          ) : (
            <>
              <AtlasHero
                regions={regions}
                records={data.records}
                forecasts={data.forecasts}
                activeRegion={activeRegion}
                activePrice={activePrice}
                activeRisk={activeRisk}
                onSelect={setActiveRegion}
                onNavigate={navigate}
              />
              <RegionStrip regions={regions} activeRegion={activeRegion} onSelect={setActiveRegion} />
              <CustomRegion loadError={data.loadError} onAdd={data.addRegion} onAdded={setActiveRegion} />

              <section id="analytics" className="analytics-section">
                <div className="section-heading">
                  <div>
                    <div className="section-index">Evidence</div>
                    <h2>{regionLabel(activeRegion)} forecast trail</h2>
                  </div>
                  <RiskBand risk={activeRisk} />
                </div>
                <ForecastCharts
                  records={activeRecords}
                  forecasts={activeForecasts}
                  activePrice={activePrice}
                  activeRisk={activeRisk}
                />
                <RiskMap regions={regions} records={data.records} forecasts={data.forecasts} activeRisk={activeRisk} />
                <ModelComparison results={data.metrics} />
                <Provenance />
              </section>

              <section id="simulator" className="simulator-section">
                <div className="section-heading">
                  <div>
                    <div className="section-index">Scenario lab</div>
                    <h2>Move the conditions</h2>
                  </div>
                  <SlidersHorizontal size={22} />
                </div>
                <Simulator
                  region={activeRegion}
                  records={activeRecords}
                  apiAvailable={data.apiAvailable}
                  onApiFailure={data.markApiUnavailable}
                />
                <Briefing region={activeRegion} forecast={outlook} price={activePrice} risk={activeRisk} />
              </section>
            </>
          )}
        </main>
      </div>
    </GlowCursorView>
  )
}

export default App
