import { useState, useTransition } from "react"
import { toast } from "sonner"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { cn } from "@/lib/utils"
import type { Tokenizer } from "@/lib/types"

/** Causal self-attention visualisation for the transformer.
 *  Calls POST /api/attention and renders each layer's per-head matrix as a
 *  CSS-grid heatmap. Lime for high attention, bone for near-zero — same
 *  palette as the rest of the UI. */

interface AttentionResponse {
  tokens: string[]
  attentions: number[][][][]   // [layer][head][q][k]
  n_layers: number
  n_heads: number
  seq_len: number
}

async function fetchAttention(
  text: string,
  tokenizer: Tokenizer,
): Promise<AttentionResponse> {
  const res = await fetch("/api/attention", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, tokenizer }),
  })
  if (!res.ok) {
    const detail = (await res.json().catch(() => ({}))).detail ?? res.statusText
    throw new Error(`${res.status}: ${detail}`)
  }
  return res.json()
}

export default function Attention() {
  const [text, setText] = useState("machine learning is a subset of")
  const [tokenizer, setTokenizer] = useState<Tokenizer>("word")
  const [data, setData] = useState<AttentionResponse | null>(null)
  const [pending, startTransition] = useTransition()

  const run = () => {
    startTransition(() => {
      void (async () => {
        try {
          setData(await fetchAttention(text, tokenizer))
        } catch (e) {
          toast.error("Attention fetch failed", {
            description: (e as Error).message,
          })
        }
      })()
    })
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Attention</h1>
        <p className="text-muted-foreground">
          Causal self-attention weights from the decoder-only transformer.
          Each cell shows how much query-token i (row) attends to key-token j
          (column). Rows sum to 1. The upper-right triangle is zero because
          the mask is causal.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Prompt</CardTitle>
          <CardDescription>
            Keep it short (≤ 64 tokens). The transformer must be trained;
            point <code className="text-xs">AUTOCOMPLETE_TRANSFORMER_CHECKPOINT_WORD</code>
            at a saved bundle to avoid lazy-fit latency.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="att-text">Text</Label>
            <Input
              id="att-text"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>
          <div className="grid grid-cols-2 gap-4 max-w-md">
            <div>
              <Label>Tokenizer</Label>
              <Select
                value={tokenizer}
                onValueChange={(v) => setTokenizer(v as Tokenizer)}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="word">word</SelectItem>
                  <SelectItem value="bpe">bpe (subword)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end">
              <Button
                onClick={run}
                disabled={pending || !text.trim()}
                className="bg-primary text-primary-foreground hover:bg-primary/90"
              >
                {pending ? "Computing…" : "Visualise"}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {pending && <Skeleton className="h-64 w-full" />}

      {data && data.attentions.length > 0 && (
        <div className="space-y-6">
          <div className="flex gap-2 flex-wrap">
            <Badge variant="secondary">layers: {data.n_layers}</Badge>
            <Badge variant="secondary">heads: {data.n_heads}</Badge>
            <Badge className="bg-accent text-accent-foreground hover:bg-accent">
              seq_len: {data.seq_len}
            </Badge>
          </div>

          {data.attentions.map((layer, li) => (
            <Card key={li}>
              <CardHeader>
                <CardTitle className="text-lg">Layer {li + 1}</CardTitle>
                <CardDescription>
                  One heatmap per head. Darker lime = stronger attention.
                </CardDescription>
              </CardHeader>
              <CardContent className="overflow-x-auto">
                <div className="flex gap-6 flex-wrap">
                  {layer.map((headMatrix, hi) => (
                    <Heatmap
                      key={hi}
                      title={`head ${hi + 1}`}
                      matrix={headMatrix}
                      tokens={data.tokens}
                    />
                  ))}
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}

/** Tiny CSS-grid heatmap. No chart library — this is just <div>s. */
function Heatmap({
  title,
  matrix,
  tokens,
}: {
  title: string
  matrix: number[][]
  tokens: string[]
}) {
  const T = matrix.length
  // Row-max normalisation so near-zero rows still show structure.
  const rowMax = matrix.map((row) => Math.max(1e-9, ...row))
  const cell = 18 // px per cell

  return (
    <div className="text-xs">
      <div className="font-mono mb-1 text-muted-foreground">{title}</div>
      <div className="flex">
        {/* Row labels (query tokens) */}
        <div
          className="grid"
          style={{
            gridTemplateRows: `repeat(${T}, ${cell}px)`,
          }}
        >
          {tokens.map((t, i) => (
            <div
              key={i}
              className="pr-2 text-right font-mono text-[10px] truncate text-muted-foreground"
              style={{ width: 56, lineHeight: `${cell}px` }}
              title={t}
            >
              {t}
            </div>
          ))}
        </div>

        {/* Matrix cells */}
        <div
          className="grid border border-border"
          style={{
            gridTemplateColumns: `repeat(${T}, ${cell}px)`,
            gridTemplateRows: `repeat(${T}, ${cell}px)`,
          }}
        >
          {matrix.flatMap((row, r) =>
            row.map((v, c) => {
              const norm = v / rowMax[r]
              // Interpolate alpha on brand lime.
              return (
                <div
                  key={`${r}-${c}`}
                  className={cn("border-r border-b border-border/40")}
                  title={`q=${tokens[r]} k=${tokens[c]}  →  ${v.toFixed(3)}`}
                  style={{
                    background: `rgba(180, 197, 64, ${norm.toFixed(3)})`,
                  }}
                />
              )
            }),
          )}
        </div>
      </div>
    </div>
  )
}
