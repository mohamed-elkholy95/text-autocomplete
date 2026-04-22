// Tiny typed fetch wrapper. All calls go through /api, which Vite proxies
// to the FastAPI on :8010 in dev. In production, the built bundle can
// either be served by FastAPI directly (then fetch("/foo")) or fronted by
// a reverse proxy that maps /api/* to the same origin.

import type {
  AutocompleteRequest,
  AutocompleteResponse,
  HealthResponse,
  JsonMetrics,
  ModelsResponse,
  VocabStats,
} from "@/lib/types"

const BASE = "/api"

async function request<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  })
  if (!res.ok) {
    let detail: string
    try {
      const body = await res.json()
      detail = body.detail ?? JSON.stringify(body)
    } catch {
      detail = await res.text()
    }
    throw new Error(`${res.status} ${res.statusText}: ${detail}`)
  }
  return res.json() as Promise<T>
}

export const api = {
  health: () => request<HealthResponse>("/health"),
  models: () => request<ModelsResponse>("/models"),
  vocab: () => request<VocabStats>("/vocab/stats"),
  metrics: () => request<JsonMetrics>("/metrics"),
  // Prometheus exposition is plain text, not JSON.
  async metricsProm(): Promise<string> {
    const res = await fetch(`${BASE}/metrics/prom`)
    if (!res.ok) throw new Error(`metrics/prom: ${res.status}`)
    return res.text()
  },
  autocomplete: (body: AutocompleteRequest) =>
    request<AutocompleteResponse>("/autocomplete", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  evalSummary: () =>
    request<{
      rows: {
        model: string
        perplexity: number
        top1: number
        top5: number
      }[]
      test_tokens: number
    }>("/eval/summary"),
}
