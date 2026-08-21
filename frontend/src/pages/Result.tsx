import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { AnalyticsPanel } from '../components/AnalyticsPanel'
import { Composer } from '../components/Composer'
import { askText, fetchMetrics } from '../api'
import { loadResult, saveResult } from '../session'
import type { AskResult, BenchLatency, BenchStatus, Metrics } from '../types'

type Panel = 'retrieved' | 'analytics'

function stageMs(data: AskResult, name: string) {
  return data.timings.filter((t) => t.name === name || t.name.startsWith(`${name}:`)).reduce((s, t) => s + t.ms, 0)
}

export function Result() {
  const nav = useNavigate()
  const [data, setData] = useState<AskResult | null>(null)
  const [q, setQ] = useState('')
  const [sttMs, setSttMs] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [bench, setBench] = useState<BenchLatency | null>(null)
  const [live, setLive] = useState<BenchLatency | null>(null)
  const [stages, setStages] = useState<Record<string, BenchLatency>>({})
  const [benchStatus, setBenchStatus] = useState<BenchStatus>('idle')
  const [benchN, setBenchN] = useState(120)
  const [panel, setPanel] = useState<Panel>('retrieved')

  function applyMetrics(m: Metrics) {
    setBench(m.bench?.latency ?? null)
    setLive(m.live ?? null)
    setStages(m.bench?.stages ?? {})
    setBenchStatus(m.bench_status ?? 'idle')
    if (m.bench_n) setBenchN(m.bench_n)
  }

  useEffect(() => {
    const stored = loadResult()
    if (!stored) {
      nav('/')
      return
    }
    setData(stored)
    setQ('')
    let stop = false
    let timer = 0
    const pull = () => {
      fetchMetrics()
        .then((m) => {
          if (stop) return
          applyMetrics(m)
          const pending = m.bench_status === 'running' || m.bench_status === 'idle'
          if (pending && !m.bench?.latency) {
            timer = window.setTimeout(pull, 800)
          }
        })
        .catch(() => undefined)
    }
    pull()
    return () => {
      stop = true
      window.clearTimeout(timer)
    }
  }, [nav])

  async function runText(query: string) {
    const text = query.trim()
    if (!text) return
    setBusy(true)
    setErr('')
    try {
      const result = await askText(text, sttMs)
      saveResult(result)
      setData(result)
      setSttMs(null)
      setQ('')
      fetchMetrics().then(applyMetrics).catch(() => undefined)
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'failed')
    } finally {
      setBusy(false)
    }
  }

  if (!data) return null
  const stt = data.stt_ms ?? 0
  const retrieval = stageMs(data, 'retrieve')
  const guardrail = stageMs(data, 'safety_check') + stageMs(data, 'ground_check')
  const generation = stageMs(data, 'extract_answer') + stageMs(data, 'generate_answer')
  const e2e = stt + data.total_ms
  const grounded = data.guardrails.find((g) => g.stage === 'grounding')
  const inputG = data.guardrails.find((g) => g.stage === 'input')
  const retG = data.guardrails.find((g) => g.stage === 'retrieval')

  return (
    <div className="result-shell">
      <aside className="result-rail" aria-label="Result pages">
        <Link to="/" className="rail-mark">
          HH
          <span>GOA</span>
        </Link>
        <nav>
          <button
            type="button"
            className={panel === 'retrieved' ? 'on' : ''}
            onClick={() => setPanel('retrieved')}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <rect x="3" y="3" width="14" height="3.4" rx="1" fill="currentColor" />
              <rect x="3" y="8.3" width="14" height="3.4" rx="1" fill="currentColor" opacity="0.7" />
              <rect x="3" y="13.6" width="9" height="3.4" rx="1" fill="currentColor" opacity="0.4" />
            </svg>
            Retrieved
          </button>
          <button
            type="button"
            className={panel === 'analytics' ? 'on' : ''}
            onClick={() => setPanel('analytics')}
          >
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" aria-hidden="true">
              <path d="M3 15a7 7 0 1 1 14 0" stroke="currentColor" strokeWidth="1.7" />
              <path d="M10 15L14.6 8.2" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" />
              <circle cx="10" cy="15" r="1.5" fill="currentColor" />
            </svg>
            Analytics
          </button>
        </nav>
      </aside>

      <div className={`result-page ${panel === 'analytics' ? 'is-analytics' : ''}`}>
        {panel === 'retrieved' && (
          <>
            <header className="nav result-nav">
              <p className="page-kicker">Retrieved</p>
              <div className={`status ${data.sla_ok ? 'ok' : ''}`}>
                <span className="dot" />
                {data.sla_ok ? 'under 200 ms' : `${data.total_ms.toFixed(0)} ms`}
              </div>
            </header>
            <p className="question">{data.transcript || data.query}</p>
            <h1 className="answer">{busy ? 'retrieving…' : data.answer.text}</h1>
            <div className="meta">
              <span className="pill">{data.detected_language}</span>
              <span className="pill">{data.query_type}</span>
              <span className="pill">{data.answer.mode}</span>
              <span className={`pill ${data.sla_ok ? 'ok' : 'bad'}`}>{data.total_ms.toFixed(1)} ms</span>
              {data.stt_ms != null && data.stt_ms > 0 && (
                <span className="pill">STT {data.stt_ms.toFixed(0)} ms</span>
              )}
              {data.answer.refused && <span className="pill bad">{data.answer.refusal_reason}</span>}
            </div>
          </>
        )}

        <div className="result-main" key={panel}>
          {panel === 'retrieved' ? (
            <div className="grid retrieved-grid">
              <article className="card">
                <h3>Passages</h3>
                {data.hits.slice(0, 4).map((h) => (
                  <div className="cite" key={`${h.rank}-${h.origin}`}>
                    {h.parent_text}
                    <small>
                      {h.chunk.strategy} · {h.origin} · {h.score.toFixed(3)}
                    </small>
                  </div>
                ))}
                {!data.hits.length && <p className="wait">none</p>}
              </article>
              <article className="card">
                <h3>Guardrail audit</h3>
                <div className="guard">
                  grounding
                  <small>
                    {grounded
                      ? grounded.allowed
                        ? 'pass · answer supported by passages'
                        : `fail · ${grounded.reason}`
                      : 'skipped'}
                  </small>
                </div>
                <div className="guard">
                  hallucination
                  <small>
                    {grounded && !grounded.allowed && grounded.categories.includes('hallucination')
                      ? 'blocked · not in retrieved context'
                      : 'clear · no ungrounded claims emitted'}
                  </small>
                </div>
                <div className="guard">
                  input
                  <small>{inputG ? `${inputG.allowed ? 'pass' : 'block'} · ${inputG.reason}` : '—'}</small>
                </div>
                <div className="guard">
                  retrieval
                  <small>{retG ? `${retG.allowed ? 'pass' : 'block'} · ${retG.reason}` : '—'}</small>
                </div>
              </article>
            </div>
          ) : (
            <AnalyticsPanel
              data={data}
              bench={bench}
              stages={stages}
              live={live}
              benchStatus={benchStatus}
              benchN={benchN}
              retrieval={retrieval}
              guardrail={guardrail}
              generation={generation}
              stt={stt}
              e2e={e2e}
            />
          )}
        </div>
        {err && <p className="err">{err}</p>}
      </div>

      {panel === 'retrieved' && (
        <div className="dock">
          <Composer
            value={q}
            onChange={setQ}
            onStt={setSttMs}
            onSubmit={() => runText(q)}
            busy={busy}
            dock
            canVoice
          />
          <div className="dock-spacer" aria-hidden="true" />
        </div>
      )}
    </div>
  )
}
