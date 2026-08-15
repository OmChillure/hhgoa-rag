import type { AskResult } from './types'

const KEY = 'vaani:last'

export function saveResult(r: AskResult) {
  sessionStorage.setItem(KEY, JSON.stringify(r))
}

export function loadResult(): AskResult | null {
  const raw = sessionStorage.getItem(KEY)
  if (!raw) return null
  try {
    return JSON.parse(raw) as AskResult
  } catch {
    return null
  }
}
