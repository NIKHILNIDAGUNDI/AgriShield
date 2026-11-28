import { Globe2 } from 'lucide-react'
import { regionLabel } from '../../lib/format'

type Props = { regions: string[]; activeRegion: string; onSelect: (region: string) => void }

export function RegionStrip({ regions, activeRegion, onSelect }: Props) {
  return (
    <section className="region-strip" aria-label="Region selector">
      {regions.map((region) => (
        <button key={region} className={activeRegion === region ? 'active' : ''} onClick={() => onSelect(region)}>
          <Globe2 size={16} />
          {regionLabel(region)}
        </button>
      ))}
    </section>
  )
}
