import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Composer } from '../components/Composer'
import { askText, fetchHealth, fetchMetrics } from '../api'
import { loadResult, saveResult } from '../session'
import type { AskResult, BenchLatency, Health, Mode } from '../types'

function stageMs(data: AskResult, name: string) {
  return data.timings.filter((t) => t.name === name || t.name.startsWith(`${name}:`)).reduce((s, t) => s + t.ms, 0)
}

export function Result() {
  const nav = useNavigate()
  const [data, setData] = useState<AskResult | null>(null)
  const [q, setQ] = useState('')
  const [mode, setMode] = useState<Mode>('fast')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [health, setHealth] = useState<Health | null>(null)
  const [bench, setBench] = useState<BenchLatency | null>(null)

  useEffect(() => {
    const stored = loadResult()
    if (!stored) {
      nav('/')
      return
    }
    setData(stored)
    setQ('')
    fetchHealth().then(setHealth).catch(() => undefined)
    fetchMetrics()
      .then((m) => setBench(m.bench?.latency ?? m.live ?? null))
      .catch(() => undefined)
  }, [nav])

  async function runText(query: string) {
    const text = query.trim()
    if (!text) return
    setBusy(true)
    setErr('')
    try {
      const result = await askText(text, mode)
      saveResult(result)
      setData(result)
      setQ('')
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
  const generation = stageMs(data, 'extract_answer') + stageMs(data, 'compose_answer')
  const e2e = stt + data.total_ms
  const grounded = data.guardrails.find((g) => g.stage === 'grounding')
  const inputG = data.guardrails.find((g) => g.stage === 'input')
  const retG = data.guardrails.find((g) => g.stage === 'retrieval')

  return (
    <div className="result-page">
      <header className="nav">
        <Link to="/" className="mark">
          HH GOA
        </Link>
        <div className={`status ${health?.ready ? 'ok' : ''}`}>
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
        {data.answer.refused && <span className="pill bad">{data.answer.refusal_reason}</span>}
      </div>

      <div className="grid">
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
        <div>
          <article className="card tel">
            <div className="tel-head">
              <h3>Telemetry</h3>
              <span className="tel-total">total {e2e.toFixed(1)} ms</span>
            </div>
            <div className="tel-grid">
              <div>
                <em>STT</em>
                <b>{stt ? `${stt.toFixed(1)} ms` : '—'}</b>
              </div>
              <div>
                <em>Retrieval</em>
                <b>{retrieval.toFixed(1)} ms</b>
              </div>
              <div>
                <em>Guardrail</em>
                <b>{guardrail.toFixed(1)} ms</b>
              </div>
              <div>
                <em>Generation</em>
                <b>{generation.toFixed(1)} ms</b>
              </div>
            </div>
            <p className={`tel-note ${data.sla_ok ? 'ok' : 'bad'}`}>
              pipeline {data.total_ms.toFixed(1)} ms · SLA &lt; 200 ms (STT not in SLA)
            </p>
            <div className="tel-p">
              <div>
                <em>P50 · target &lt; 200 ms</em>
                <b>{bench ? `${bench.p50_ms.toFixed(1)} ms` : '—'}</b>
              </div>
              <div>
                <em>P70</em>
                <b>{bench ? `${bench.p70_ms.toFixed(1)} ms` : '—'}</b>
              </div>
              <div>
                <em>P100</em>
                <b>{bench ? `${bench.p100_ms.toFixed(1)} ms` : '—'}</b>
              </div>
            </div>
          </article>
          <article className="card" style={{ marginTop: 14 }}>
            <h3>Guardrail audit</h3>
            <div className="guard">
              grounding
              <small>{grounded ? (grounded.allowed ? 'pass · answer supported by passages' : `fail · ${grounded.reason}`) : 'skipped'}</small>
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
      </div>
      {err && <p className="err">{err}</p>}

      <div className="dock">
        <Composer
          value={q}
          onChange={setQ}
          mode={mode}
          onMode={setMode}
          onSubmit={() => runText(q)}
          busy={busy}
          dock
          canVoice={health?.sarvam}
        />
        <div className="dock-spacer" aria-hidden="true" />
      </div>
    </div>
  )
}
