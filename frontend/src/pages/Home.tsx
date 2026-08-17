import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Composer } from '../components/Composer'
import { askText, fetchHealth, fetchSamples } from '../api'
import type { Health, Sample } from '../types'
import { saveResult } from '../session'

export function Home() {
  const nav = useNavigate()
  const [q, setQ] = useState('')
  const [sttMs, setSttMs] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')
  const [health, setHealth] = useState<Health | null>(null)
  const [samples, setSamples] = useState<Sample[]>([])

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => setHealth(null))
    fetchSamples().then(setSamples).catch(() => undefined)
  }, [])

  async function runText(query: string) {
    const text = query.trim()
    if (!text) return
    setBusy(true)
    setErr('')
    try {
      const result = await askText(text, sttMs)
      saveResult(result)
      setSttMs(null)
      nav('/result')
    } catch (e) {
      setErr(e instanceof Error ? e.message : 'failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <header className="nav">
        <span className="mark">HH GOA</span>
        <div className={`status ${health?.ready ? 'ok' : ''}`}>
          <span className="dot" />
          {health?.ready ? `${health.stats.chunks ?? 0} chunks · live` : 'index offline'}
        </div>
      </header>
      <section className="home">
        <div className="stage">
          <span className="badge">hh_goa</span>
          <h1 className="word">Vaani</h1>
          <p className="tag">Goa, India · 28 – 31 Oct · voice rag</p>
          <Composer
            value={q}
            onChange={setQ}
            onStt={setSttMs}
            onSubmit={() => runText(q)}
            busy={busy}
            canVoice={health?.sarvam}
          />
          {busy && <p className="wait">retrieving…</p>}
          {err && <p className="err">{err}</p>}
          <div className="chips">
            {samples.slice(0, 5).map((s) => (
              <button key={s.query} type="button" onClick={() => runText(s.query)}>
                {s.query}
              </button>
            ))}
          </div>
        </div>
      </section>
    </>
  )
}
