import type { AskResult, BenchLatency, BenchStatus } from '../types'
import { Gauge } from './Gauge'

type Props = {
  data: AskResult
  bench: BenchLatency | null
  stages: Record<string, BenchLatency>
  live: BenchLatency | null
  benchStatus?: BenchStatus
  benchN?: number
  retrieval: number
  guardrail: number
  generation: number
  stt: number
  e2e: number
}

function Bar({ label, ms, max }: { label: string; ms: number; max: number }) {
  const w = max > 0 ? Math.max(3, Math.min(100, (ms / max) * 100)) : 0
  return (
    <div className="bar">
      <span>{label}</span>
      <i style={{ width: `${w}%` }} />
      <b>{ms.toFixed(1)} ms</b>
    </div>
  )
}

export function AnalyticsPanel({
  data,
  bench,
  stages,
  live,
  benchStatus = 'idle',
  benchN = 120,
  retrieval,
  guardrail,
  generation,
  stt,
  e2e,
}: Props) {
  const measuring = benchStatus === 'running' || (benchStatus === 'idle' && !bench)
  const lat = bench
  const n = lat?.n ?? 0
  const hit = lat?.under_200ms_pct
  const thisMax = Math.max(data.total_ms, stt, retrieval, 1)
  const retrieveBench = stages.retrieve

  return (
    <div className="analytics">
      <article className="card tel">
        <div className="tel-head">
          <h3>Pipeline latency</h3>
          <span className={`tel-total ${data.sla_ok ? 'ok' : 'bad'}`}>
            this query {data.total_ms.toFixed(1)} ms
          </span>
        </div>
        <p className="tel-note">
          {measuring
            ? `Measuring P50 / P70 / P100 over ${benchN} test queries…`
            : `P50 / P70 / P100 across ${n || '—'} test queries — not this one ask · SLA < 200 ms`}
        </p>
        <div className="gauges">
          <Gauge
            label="P50"
            value={lat ? lat.p50_ms : null}
            hint="median of the sweep"
          />
          <Gauge
            label="P70"
            value={lat ? lat.p70_ms : null}
            hint="70th percentile"
          />
          <Gauge
            label="P100"
            value={lat ? lat.p100_ms : null}
            hint="slowest in the sweep"
          />
        </div>
        {lat && (
          <p className="tel-note">
            {lat.p90_ms != null ? `P90 ${lat.p90_ms.toFixed(1)} ms` : ''}
            {lat.mean_ms != null ? ` · mean ${lat.mean_ms.toFixed(1)} ms` : ''}
            {hit != null ? ` · ${hit.toFixed(0)}% under 200 ms` : ''}
          </p>
        )}
        {benchStatus === 'error' && !lat && (
          <p className="tel-note bad">sweep failed — gauges need the index query table</p>
        )}
      </article>

      <div className="analytics-split">
        <article className="card">
          <h3>This query</h3>
          <div className="gauges gauges-sm">
            <Gauge label="Pipeline" value={data.total_ms} hint="chunk + retrieve + answer" />
            <Gauge label="End to end" value={e2e} hint="includes STT if used" />
          </div>
          <div className="bars">
            {stt > 0 && <Bar label="STT" ms={stt} max={thisMax} />}
            <Bar label="Retrieve" ms={retrieval} max={thisMax} />
            <Bar label="Guard" ms={guardrail} max={thisMax} />
            <Bar label="Answer" ms={generation} max={thisMax} />
          </div>
        </article>

        <article className="card">
          <h3>Retrieve stage (same sweep)</h3>
          {retrieveBench ? (
            <>
              <div className="gauges gauges-sm">
                <Gauge label="P50 retrieve" value={retrieveBench.p50_ms} />
                <Gauge label="P70 retrieve" value={retrieveBench.p70_ms} />
              </div>
              <div className="bars">
                <Bar label="P50" ms={retrieveBench.p50_ms} max={200} />
                <Bar label="P70" ms={retrieveBench.p70_ms} max={200} />
                {retrieveBench.p90_ms != null && (
                  <Bar label="P90" ms={retrieveBench.p90_ms} max={200} />
                )}
                <Bar label="P100" ms={retrieveBench.p100_ms} max={200} />
              </div>
            </>
          ) : measuring ? (
            <p className="wait">same {benchN} queries — retrieve percentiles incoming</p>
          ) : (
            <p className="wait">no retrieve timings in the sweep</p>
          )}
          {live && live.n > 0 && (
            <p className="tel-note" style={{ marginTop: 12 }}>
              this session {live.n} ask{live.n === 1 ? '' : 's'} · P50 {live.p50_ms.toFixed(1)} ms
            </p>
          )}
        </article>
      </div>
    </div>
  )
}
