// Types re-exported from the generated OpenAPI schema so the frontend and
// FastAPI stay in lockstep. Regenerate with:
//
//   # (API must be running on :8010)
//   npm run gen:api
//
// which fetches /openapi.json and writes src/lib/api-schema.d.ts. Do NOT
// hand-edit api-schema.d.ts — it's regenerated on every run.

import type { components } from "./api-schema"

export type ModelId = "ngram" | "markov" | "lstm" | "transformer"
export type Tokenizer = "word" | "bpe"

// Pydantic models -> TypeScript aliases. Pulling from the generated
// schemas keeps every field (including optional ones like tokenizer_name)
// in sync automatically. Aliases use the names the rest of the app already
// imports, so this file is the only swap point.
type S = components["schemas"]

export type AutocompleteRequest = S["AutocompleteRequest"]
export type AutocompleteResponse = S["AutocompleteResponse"]
export type Suggestion = S["Suggestion"]
export type ModelsResponse = { models: ModelInfo[] }

export interface ModelInfo {
  id: ModelId
  name: string
  description: string
  max_ngram_order?: number
}

export interface HealthResponse {
  status: string
  version: string
}

export interface VocabStats {
  corpus: {
    total_tokens: number
    unique_tokens: number
    avg_sentence_length: number
    [k: string]: unknown
  }
  ngram_vocab_size: number
  markov_vocab_size: number
  available_models: ModelId[]
}

export interface JsonMetrics {
  total_requests: number
  rate_limited: number
  endpoints: Record<string, number>
}
