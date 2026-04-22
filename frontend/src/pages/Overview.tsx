import { useEffect, useState } from "react"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import type { ModelInfo, VocabStats } from "@/lib/types"

/** Overview page — model catalogue + corpus stats. */
export default function Overview() {
  const [models, setModels] = useState<ModelInfo[] | null>(null)
  const [stats, setStats] = useState<VocabStats | null>(null)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([api.models(), api.vocab()])
      .then(([m, v]) => {
        setModels(m.models)
        setStats(v)
      })
      .catch((e) => setErr(String(e)))
  }, [])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight text-foreground">
          Overview
        </h1>
        <p className="text-muted-foreground">
          Models exposed by the API and a snapshot of the training corpus.
        </p>
      </div>

      {err && (
        <Card className="border-destructive">
          <CardContent className="pt-6 text-destructive">{err}</CardContent>
        </Card>
      )}

      {/* Corpus stats */}
      <Card>
        <CardHeader>
          <CardTitle>Corpus</CardTitle>
          <CardDescription>
            Built-in sample corpus used for lazy model training.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <Stat label="Total tokens" value={stats?.corpus.total_tokens} />
          <Stat label="Unique tokens" value={stats?.corpus.unique_tokens} />
          <Stat
            label="N-gram vocab"
            value={stats?.ngram_vocab_size}
          />
          <Stat label="Markov vocab" value={stats?.markov_vocab_size} />
        </CardContent>
      </Card>

      {/* Model cards */}
      <div>
        <h2 className="text-xl font-semibold mb-3 text-foreground">Models</h2>
        {models === null ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-40 w-full" />
            <Skeleton className="h-40 w-full" />
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {models.map((m) => (
              <Card key={m.id}>
                <CardHeader className="flex flex-row items-center justify-between">
                  <div>
                    <CardTitle className="capitalize">{m.name}</CardTitle>
                    <CardDescription className="font-mono text-xs mt-1">
                      id: {m.id}
                    </CardDescription>
                  </div>
                  <Badge className="bg-accent text-accent-foreground hover:bg-accent">
                    available
                  </Badge>
                </CardHeader>
                <CardContent className="text-sm leading-relaxed">
                  {m.description}
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function Stat({ label, value }: { label: string; value: number | undefined }) {
  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-muted-foreground">
        {label}
      </div>
      <div className="text-2xl font-semibold text-primary tabular-nums">
        {value === undefined ? "—" : value.toLocaleString()}
      </div>
    </div>
  )
}
