export function Provenance() {
  return (
    <section className="provenance">
      <div>
        <span className="provenance-dot observed" />
        Observed
      </div>
      <div>
        <span className="provenance-dot estimated" />
        Estimated fallback
      </div>
      <div>
        <span className="provenance-dot synthetic" />
        Synthetic
      </div>
    </section>
  )
}
