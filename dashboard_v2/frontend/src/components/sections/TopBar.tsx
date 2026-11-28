import { Wheat } from 'lucide-react'
import { views, type View } from '../../lib/views'
import { GooeyNavView } from '../reactBits'

const navItems = views.map((view) => ({ label: view[0].toUpperCase() + view.slice(1), href: `#${view}` }))

type Props = {
  activeView: View
  onNavigate: (view: View) => void
  apiAvailable: boolean
  modelVersion: string | null
}

export function TopBar({ activeView, onNavigate, apiAvailable, modelVersion }: Props) {
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand-mark"><Wheat size={18} /></span>
        <span>AgriShield</span>
      </div>
      <GooeyNavView
        items={navItems}
        theme="light"
        color="#69c74f"
        activeTextColor="#1f2a25"
        activeIndex={views.indexOf(activeView)}
        onChange={(index: number) => onNavigate(views[index])}
      />
      <div className={`api-state ${apiAvailable ? 'online' : 'fallback'}`} role="status" aria-live="polite">
        <span />
        {apiAvailable ? `API · ${modelVersion ?? 'connected'}` : 'Static fallback'}
      </div>
    </header>
  )
}
