import type { AskResult, Health, Metrics, Mode, Sample } from './types'

export async function fetchHealth(): Promise<Health> {
  const r = await fetch('/api/health')
  if (!r.ok) throw new Error('health failed')
  return r.json()
}

export async function fetchMetrics(): Promise<Metrics> {
  const r = await fetch('/api/metrics')
  if (!r.ok) throw new Error('metrics failed')
  return r.json()
}

export async function fetchSamples(): Promise<Sample[]> {
  const r = await fetch('/api/samples')
  if (!r.ok) return []
  const data = await r.json()
  return data.queries ?? []
}

export async function askText(query: string, mode: Mode): Promise<AskResult> {
  const r = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, mode }),
  })
  const data = await r.json()
  if (!r.ok) throw new Error(data.detail || 'ask failed')
  return data
}

function writeStr(view: DataView, offset: number, str: string) {
  for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i))
}

function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const n = samples.length
  const buf = new ArrayBuffer(44 + n * 2)
  const view = new DataView(buf)
  writeStr(view, 0, 'RIFF')
  view.setUint32(4, 36 + n * 2, true)
  writeStr(view, 8, 'WAVE')
  writeStr(view, 12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeStr(view, 36, 'data')
  view.setUint32(40, n * 2, true)
  let o = 44
  for (let i = 0; i < n; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buf], { type: 'audio/wav' })
}

export async function blobToWav(blob: Blob): Promise<Blob> {
  const ctx = new AudioContext()
  try {
    const raw = await blob.arrayBuffer()
    const decoded = await ctx.decodeAudioData(raw.slice(0))
    const rate = 16000
    const frames = Math.max(1, Math.ceil(decoded.duration * rate))
    const offline = new OfflineAudioContext(1, frames, rate)
    const src = offline.createBufferSource()
    src.buffer = decoded
    src.connect(offline.destination)
    src.start(0)
    const rendered = await offline.startRendering()
    return encodeWav(rendered.getChannelData(0), rate)
  } finally {
    await ctx.close().catch(() => undefined)
  }
}

export async function transcribeAudio(blob: Blob): Promise<{ text: string; ms: number }> {
  const wav = await blobToWav(blob)
  const fd = new FormData()
  fd.append('file', wav, 'clip.wav')
  const r = await fetch('/api/transcribe', { method: 'POST', body: fd })
  const data = await r.json()
  if (!r.ok) throw new Error(data.detail || 'transcribe failed')
  return { text: data.text || '', ms: data.ms ?? 0 }
}
