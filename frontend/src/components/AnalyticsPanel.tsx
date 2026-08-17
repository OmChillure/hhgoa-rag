import type { AskResult, BenchLatency } from '../types'
import { Gauge } from './Gauge'

type Props = {
  data: AskResult
  bench: BenchLatency | null
  stages: Record<string, BenchLatency>
  live: BenchLatency | null
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
  retrieval,
  guardrail,
  generation,
  stt,
  e2e,
}: Props) {
  const lat = bench ?? live
  const n = lat?.n ?? 0
  const hit = lat?.under_200ms_pct
  const thisMax = Math.max(data.total_ms, stt, retrieval, 1)
  const retrieveBench = stages.retrieve

  return (
    <div className="analytics">
      <article className="card tel">
        <div className="tel-head">
          <h3>Retriever latency</h3>
          <span className={`tel-total ${data.sla_ok ? 'ok' : 'bad'}`}>
            this query {data.total_ms.toFixed(1)} ms
          </span>
        </div>
        <p className="tel-note">
          Bench P50 / P70 / P100 over {n || '—'} queries · SLA &lt; 200 ms (STT not in SLA)
        </p>
        <div className="gauges">
          <Gauge
            label="P50"
            value={lat ? lat.p50_ms : null}
            hint="median pipeline"
          />
          <Gauge
            label="P70"
            value={lat ? lat.p70_ms : null}
            hint="70th percentile"
          />
          <Gauge
            label="P100"
            value={lat ? lat.p100_ms : null}
            hint="slowest query"
          />
        </div>
        {lat && (
          <p className="tel-note">
            {lat.p90_ms != null ? `P90 ${lat.p90_ms.toFixed(1)} ms` : ''}
            {lat.mean_ms != null ? ` · mean ${lat.mean_ms.toFixed(1)} ms` : ''}
            {hit != null ? ` · ${hit.toFixed(0)}% under 200 ms` : ''}
          </p>
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
          <h3>Retrieve stage (bench)</h3>
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
          ) : (
            <p className="wait">no bench file — run scripts/bench.py</p>
          )}
          {live && live.n > 0 && (
            <p className="tel-note" style={{ marginTop: 12 }}>
              live session P50 {live.p50_ms.toFixed(1)} ms · {live.n} asks
            </p>
          )}
        </article>
      </div>
    </div>
  )
}
