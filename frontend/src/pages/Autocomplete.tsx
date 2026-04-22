import { useActionState, useEffect, useState } from "react"
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
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import type {
  AutocompleteResponse,
  ModelId,
  ModelsResponse,
  Tokenizer,
} from "@/lib/types"
import { cn } from "@/lib/utils"

type ActionState =
  | { status: "idle" }
  | { status: "ok"; data: AutocompleteResponse; tookMs: number }
  | { status: "error"; error: string }

/** Autocomplete page — form + top-k suggestions. */
export default function Autocomplete() {
  const [availableModels, setAvailable] = useState<ModelId[]>([
    "ngram",
    "markov",
  ])
  const [model, setModel] = useState<ModelId>("ngram")
  const [tokenizer, setTokenizer] = useState<Tokenizer>("word")
  const [topK, setTopK] = useState(5)

  useEffect(() => {
    api
      .models()
      .then((r: ModelsResponse) =>
        setAvailable(r.models.map((m) => m.id)),
      )
      .catch(() => {
        /* sidebar health pill covers the error state */
      })
  }, [])

  // React 19's useActionState is ideal for form submissions: it handles the
  // pending state, lets you return a typed result, and plays nicely with
  // progressive enhancement (though we're SPA-only here).
  const [state, submit, pending] = useActionState<ActionState, FormData>(
    async (_prev, formData) => {
      const text = (formData.get("text") as string)?.trim() ?? ""
      if (!text) {
        toast.warning("Enter some text first.")
        return { status: "error", error: "Enter some text first." }
      }
      const t0 = performance.now()
      try {
        const data = await api.autocomplete({
          text,
          top_k: topK,
          model,
          tokenizer:
            model === "ngram" || model === "markov" ? "word" : tokenizer,
        })
        return { status: "ok", data, tookMs: performance.now() - t0 }
      } catch (e) {
        const msg = (e as Error).message
        toast.error("Autocomplete failed", { description: msg })
        return { status: "error", error: msg }
      }
    },
    { status: "idle" },
  )

  const isNeural = model === "lstm" || model === "transformer"

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight">Autocomplete</h1>
        <p className="text-muted-foreground">
          Send text to <code className="text-xs">/autocomplete</code> and
          inspect the top-k next-word suggestions.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Request</CardTitle>
          <CardDescription>Fill in the fields and hit Predict.</CardDescription>
        </CardHeader>
        <CardContent>
          <form action={submit} className="space-y-4">
            <div>
              <Label htmlFor="text">Text</Label>
              <Input
                id="text"
                name="text"
                defaultValue="machine learning is a"
                placeholder="Type a few words…"
                autoFocus
              />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <Label>Model</Label>
                <Select
                  value={model}
                  onValueChange={(v) => setModel(v as ModelId)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {(["ngram", "markov", "lstm", "transformer"] as ModelId[])
                      .filter((m) => availableModels.includes(m))
                      .map((m) => (
                        <SelectItem key={m} value={m}>
                          {m}
                        </SelectItem>
                      ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <Label>Tokenizer</Label>
                <Select
                  value={tokenizer}
                  onValueChange={(v) => setTokenizer(v as Tokenizer)}
                  disabled={!isNeural}
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

              <div>
                <Label htmlFor="topk">Top-K</Label>
                <Input
                  id="topk"
                  type="number"
                  min={1}
                  max={20}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                />
              </div>
            </div>

            <Button
              type="submit"
              disabled={pending}
              className="bg-primary text-primary-foreground hover:bg-primary/90"
            >
              {pending ? "Predicting…" : "Predict"}
            </Button>
          </form>
        </CardContent>
      </Card>

      {state.status === "error" && (
        <Card className="border-destructive">
          <CardContent className="pt-6 text-destructive text-sm">
            {state.error}
          </CardContent>
        </Card>
      )}

      {state.status === "ok" && (
        <Card>
          <CardHeader className="flex flex-row items-start justify-between">
            <div>
              <CardTitle>Suggestions</CardTitle>
              <CardDescription className="mt-1">
                Context: <code className="text-xs">{state.data.context}</code>
              </CardDescription>
            </div>
            <div className="flex gap-2">
              <Badge variant="secondary">model: {state.data.model}</Badge>
              <Badge className="bg-accent text-accent-foreground hover:bg-accent">
                {state.tookMs.toFixed(0)} ms
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-12">#</TableHead>
                  <TableHead>Word</TableHead>
                  <TableHead className="text-right">Probability</TableHead>
                  <TableHead>Bar</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {state.data.suggestions.map((s, i) => {
                  const max = state.data.suggestions[0]?.probability ?? 1
                  const pct = Math.max(1, (s.probability / max) * 100)
                  return (
                    <TableRow
                      key={`${s.word}-${i}`}
                      className={cn(i === 0 && "bg-accent/10")}
                    >
                      <TableCell className="font-mono">{i + 1}</TableCell>
                      <TableCell className="font-medium">
                        <code>{s.word}</code>
                      </TableCell>
                      <TableCell className="text-right tabular-nums">
                        {s.probability.toFixed(4)}
                      </TableCell>
                      <TableCell className="w-56">
                        <div
                          className={cn(
                            "h-2 rounded-sm",
                            i === 0 ? "bg-accent" : "bg-primary",
                          )}
                          style={{ width: `${pct}%` }}
                        />
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
