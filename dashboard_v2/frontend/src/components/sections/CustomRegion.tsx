import { useState } from 'react'
import { ChevronRight, Search } from 'lucide-react'

type Props = {
  loadError: string
  onAdd: (name: string, onStatus: (status: string) => void) => Promise<string>
  onAdded: (region: string) => void
}

export function CustomRegion({ loadError, onAdd, onAdded }: Props) {
  const [name, setName] = useState('')
  const [status, setStatus] = useState('')

  async function submit() {
    if (!name.trim()) {
      setStatus('Enter a region name to begin.')
      return
    }
    setStatus('Resolving region and climatology...')
    try {
      const region = await onAdd(name, setStatus)
      onAdded(region)
      setStatus(`Active: ${region}. Observed weather and CPI are labeled in region provenance.`)
      setName('')
    } catch (error) {
      setStatus(error instanceof Error ? error.message : 'Custom-region lookup failed.')
    }
  }

  return (
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
            value={name}
            onChange={(event) => setName(event.target.value)}
            onKeyDown={(event) => { if (event.key === 'Enter') void submit() }}
            placeholder="Spain, Kenya, Tokyo..."
          />
          <button onClick={() => void submit()} aria-label="Fetch custom region forecast">
            <ChevronRight size={18} />
          </button>
        </div>
        <p className="custom-status" role="status">{status || loadError || 'Observed weather and CPI are retained as observed; generated fields are labeled.'}</p>
      </div>
    </section>
  )
}
