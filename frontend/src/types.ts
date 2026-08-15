export type Mode = 'fast' | 'quality'

export type Hit = {
  score: number
  rank: number
  origin: string
  parent_text: string
  chunk: {
    strategy: string
    language: string
    text: string
  }
}

export type Guard = {
  allowed: boolean
  stage: string
  reason: string
  categories: string[]
}

export type Timing = { name: string; ms: number }

export type AskResult = {
  query: string
  transcript?: string | null
  detected_language: string
  query_type: string
  answer: {
    text: string
    mode: string
    confidence: number
    grounded: boolean
    coverage: number
    refused: boolean
    refusal_reason: string
  }
  guardrails: Guard[]
  hits: Hit[]
  timings: Timing[]
  total_ms: number
  sla_ok: boolean
  harness_trace: { tool: string; ok: boolean; attempt: number }[]
  stt_ms?: number | null
}

export type Health = {
  ready: boolean
  sarvam: boolean
  sarvam_keys: number
  gemini: boolean
  sla_ms: number
  stats: { chunks?: number }
}

export type Sample = { query: string; hi_query?: string; query_type?: string }

export type BenchLatency = {
  n: number
  p50_ms: number
  p70_ms: number
  p100_ms: number
  under_200ms_pct: number
}

export type Metrics = {
  live: BenchLatency
  bench: { latency?: BenchLatency }
}
