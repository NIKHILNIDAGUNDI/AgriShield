export function formatNumber(value: number | null | undefined, digits = 1) {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—'
}

/** "2026-04" becomes "April 2026". */
export function monthLabel(yearMonth: string) {
  return new Date(`${yearMonth}-01T00:00:00Z`)
    .toLocaleDateString('en-US', { month: 'long', year: 'numeric', timeZone: 'UTC' })
}

export function regionLabel(region: string) {
  return region === 'US' ? 'United States' : region
}

export function riskBandName(risk: number) {
  return risk >= 65 ? 'Critical' : risk >= 30 ? 'Elevated' : 'Secure'
}
