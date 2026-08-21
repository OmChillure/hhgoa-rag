import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Composer } from '../components/Composer'
import { askText } from '../api'
import { PROMPT_EXAMPLES } from '../prompts'
import { saveResult } from '../session'

export function Home() {
  const nav = useNavigate()
  const [q, setQ] = useState('')
  const [sttMs, setSttMs] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState('')

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
        <div className="status ok">
          <span className="dot" />
          live
        </div>
      </header>
      <section className="home">
        <div className="stage">
          <span className="badge">hh_goa</span>
          <h1 className="word">Echo</h1>
          <p className="tag">Goa, India · 28 – 31 Oct · voice rag</p>
          <Composer
            value={q}
            onChange={setQ}
            onStt={setSttMs}
            onSubmit={() => runText(q)}
            busy={busy}
            canVoice
          />
          {busy && <p className="wait">retrieving…</p>}
          {err && <p className="err">{err}</p>}
          <div className="prompts">
            <p className="prompts-label">Try a language</p>
            <div className="chips">
              {PROMPT_EXAMPLES.map((p) => (
                <button
                  key={`${p.lang}-${p.text}`}
                  type="button"
                  disabled={busy}
                  onClick={() => {
                    setQ(p.text)
                    void runText(p.text)
                  }}
                >
                  <span className="chip-lang">{p.lang}</span>
                  {p.text}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>
    </>
  )
}
