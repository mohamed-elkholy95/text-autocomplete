// Mirror of the Pydantic models in src/api/main.py. Keep these in sync
// by hand — the project is small enough that codegen isn't worth it.

export type ModelId = "ngram" | "markov" | "lstm" | "transformer"
export type Tokenizer = "word" | "bpe"

export interface ModelInfo {
  id: ModelId
  name: string
  description: string
  max_ngram_order?: number
}

export interface ModelsResponse {
  models: ModelInfo[]
}

export interface Suggestion {
  word: string
  probability: number
}

export interface AutocompleteRequest {
  text: string
  top_k?: number
  model?: ModelId
  tokenizer?: Tokenizer
  tokenizer_name?: string | null
}

export interface AutocompleteResponse {
  suggestions: Suggestion[]
  context: string
  model: ModelId
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
