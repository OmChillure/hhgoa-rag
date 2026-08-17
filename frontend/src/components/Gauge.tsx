type Props = {
  label: string
  value: number | null
  max?: number
  unit?: string
  hint?: string
}

const CX = 100
const CY = 108
const R = 72
const START = -210
const SWEEP = 240

function toRad(deg: number) {
  return (deg * Math.PI) / 180
}

function pt(r: number, deg: number) {
  const a = toRad(deg)
  return { x: CX + r * Math.cos(a), y: CY + r * Math.sin(a) }
}

function arcPath(r: number, a0: number, a1: number) {
  const s = pt(r, a0)
  const e = pt(r, a1)
  const large = Math.abs(a1 - a0) > 180 ? 1 : 0
  return `M ${s.x.toFixed(2)} ${s.y.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${e.x.toFixed(2)} ${e.y.toFixed(2)}`
}

function valueAngle(value: number, max: number) {
  const t = Math.max(0, Math.min(value / max, 1))
  return START + t * SWEEP
}

export function Gauge({ label, value, max = 200, unit = 'ms', hint }: Props) {
  const v = value ?? 0
  const ok = value != null && value < max
  const needle = ok ? '#fee101' : '#e08a72'
  const angle = valueAngle(v, max)
  const tip = pt(R - 16, angle)
  const majors = [0, 0.25, 0.5, 0.75, 1]
  const minors = Array.from({ length: 21 }, (_, i) => i / 20)

  return (
    <div className={`gauge ${value == null ? '' : ok ? 'ok' : 'bad'}`}>
      <div className="gauge-wrap">
        <svg viewBox="0 0 200 150" className="gauge-svg" aria-hidden="true">
          <path d={arcPath(R, START, START + SWEEP)} className="gauge-track" />
          <path d={arcPath(R, START, START + SWEEP * 0.4)} className="gauge-zone good" />
          <path d={arcPath(R, START + SWEEP * 0.4, START + SWEEP * 0.75)} className="gauge-zone warn" />
          <path d={arcPath(R, START + SWEEP * 0.75, START + SWEEP)} className="gauge-zone hot" />
          {minors.map((t) => {
            const d = START + t * SWEEP
            const major = majors.includes(t)
            const a = pt(R - (major ? 8 : 4), d)
            const b = pt(R + 1, d)
            return (
              <line
                key={t}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                className={major ? 'gauge-tick major' : 'gauge-tick'}
              />
            )
          })}
          {majors.map((t) => {
            const d = START + t * SWEEP
            const p = pt(R - 20, d)
            return (
              <text key={t} x={p.x} y={p.y} className="gauge-tick-label" textAnchor="middle" dominantBaseline="middle">
                {Math.round(max * t)}
              </text>
            )
          })}
          {value != null && (
            <line x1={CX} y1={CY} x2={tip.x} y2={tip.y} stroke={needle} strokeWidth="2.6" strokeLinecap="round" />
          )}
          <circle cx={CX} cy={CY} r="9" fill="#0c0e0b" stroke={needle} strokeWidth="2.2" />
          <circle cx={CX} cy={CY} r="3.4" fill={needle} />
        </svg>
      </div>
      <strong>{value == null ? '—' : `${v.toFixed(1)} ${unit}`}</strong>
      <em>{label}</em>
      {hint && <small>{hint}</small>}
    </div>
  )
}
