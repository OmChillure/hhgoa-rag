import { useEffect, useRef, useState } from 'react'
import { transcribeAudio } from '../api'

type Props = {
  value: string
  onChange: (v: string) => void
  onStt?: (ms: number | null) => void
  onSubmit: () => void
  busy?: boolean
  dock?: boolean
  canVoice?: boolean
}

export function Composer({
  value,
  onChange,
  onStt,
  onSubmit,
  busy,
  dock,
  canVoice,
}: Props) {
  const [mic, setMic] = useState<'off' | 'rec' | 'stt'>('off')
  const recRef = useRef<MediaRecorder | null>(null)
  const chunks = useRef<Blob[]>([])
  const ta = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!ta.current || dock) return
    ta.current.style.height = 'auto'
    ta.current.style.height = `${Math.min(120, ta.current.scrollHeight)}px`
  }, [value, dock])

  async function finishClip(blob: Blob) {
    setMic('stt')
    try {
      const { text, ms } = await transcribeAudio(blob)
      if (text) {
        onChange(text)
        onStt?.(ms)
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'transcribe failed')
    } finally {
      setMic('off')
    }
  }

  async function toggleMic() {
    if (mic === 'stt') return
    if (mic === 'rec') {
      recRef.current?.stop()
      return
    }
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    const mime = ['audio/webm;codecs=opus', 'audio/webm'].find((t) => MediaRecorder.isTypeSupported(t))
    const rec = mime ? new MediaRecorder(stream, { mimeType: mime }) : new MediaRecorder(stream)
    chunks.current = []
    rec.ondataavailable = (e) => e.data.size && chunks.current.push(e.data)
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop())
      const blob = new Blob(chunks.current, { type: rec.mimeType || 'audio/webm' })
      if (blob.size < 800) {
        setMic('off')
        return
      }
      void finishClip(blob)
    }
    rec.start(250)
    recRef.current = rec
    setMic('rec')
  }

  const actions = (
    <div className="actions">
      <button
        type="button"
        className={`icon-btn${mic === 'rec' ? ' rec' : ''}${mic === 'stt' ? ' wait' : ''}`}
        disabled={mic === 'stt'}
        onClick={() => toggleMic().catch((err) => alert(err.message))}
        title={
          mic === 'stt'
            ? 'transcribing…'
            : mic === 'rec'
              ? 'click to stop'
              : canVoice
                ? 'voice'
                : 'add SARVAM_API_KEY for voice'
        }
        aria-label="voice"
      >
        {mic === 'stt' ? (
          <span className="spin" />
        ) : (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="9" y="3" width="6" height="11" rx="3" />
            <path d="M6 11a6 6 0 0 0 12 0M12 17v4" />
          </svg>
        )}
      </button>
      <button
        type="button"
        className="go"
        disabled={busy || mic !== 'off' || !value.trim()}
        onClick={onSubmit}
        aria-label="send"
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.3">
          <path d="M12 19V5M5 12l7-7 7 7" />
        </svg>
      </button>
    </div>
  )

  if (dock) {
    return (
      <div className="composer is-bar">
        <input
          className="inline-input"
          placeholder="Ask the corpus…"
          value={value}
          onChange={(e) => {
            onChange(e.target.value)
            onStt?.(null)
          }}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && value.trim() && !busy) onSubmit()
          }}
        />
        {actions}
      </div>
    )
  }

  return (
    <div className="composer">
      <textarea
        ref={ta}
        rows={2}
        placeholder="Ask me anything"
        value={value}
        onChange={(e) => {
          onChange(e.target.value)
          onStt?.(null)
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            if (value.trim() && !busy) onSubmit()
          }
        }}
      />
      <div className="row">
        <span className="tool-label">Search</span>
        {actions}
      </div>
    </div>
  )
}
