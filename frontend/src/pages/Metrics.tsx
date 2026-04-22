import { useEffect, useState, useTransition } from "react"
import { toast } from "sonner"
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import type { JsonMetrics } from "@/lib/types"

/** Metrics page — JSON /metrics, a per-endpoint bar chart, and a preview
 *  of the Prometheus /metrics/prom scrape endpoint. */
interface EvalRow {
  model: string
  perplexity: number
  top1: number
  top5: number
}

export default function Metrics() {
  const [metrics, setMetrics] = useState<JsonMetrics | null>(null)
  const [prom, setProm] = useState<string | null>(null)
  const [evalRows, setEvalRows] = useState<EvalRow[] | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [pending, startTransition] = useTransition()

  const refresh = () =>
    startTransition(() => {
      void (async () => {
        try {
          const [m, p, evalResp] = await Promise.all([
            api.metrics(),
            api.metricsProm(),
            api.evalSummary(),
          ])
          setMetrics(m)
          setProm(p)
          setEvalRows(evalResp.rows)
          setErr(null)
        } catch (e) {
          const msg = (e as Error).message
          setErr(msg)
          toast.error("Failed to load metrics", { description: msg })
        }
      })()
    })

  useEffect(refresh, [])

  const chartData =
    metrics && Object.entries(metrics.endpoints).map(([name, count]) => ({
      name,
      count,
    }))

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight">Metrics</h1>
          <p className="text-muted-foreground">
            Live counters from the FastAPI backend.
          </p>
        </div>
        <Button
          onClick={refresh}
          disabled={pending}
          className="bg-primary text-primary-foreground hover:bg-primary/90"
        >
          {pending ? "Refreshing…" : "Refresh"}
        </Button>
      </div>

      {err && (
        <Card className="border-destructive">
          <CardContent className="pt-6 text-destructive text-sm">
            {err}
          </CardContent>
        </Card>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardHeader>
            <CardDescription>Total requests</CardDescription>
            <CardTitle className="text-3xl text-primary tabular-nums">
              {metrics?.total_requests ?? "—"}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader>
            <CardDescription>Rate limited (429)</CardDescription>
            <CardTitle className="text-3xl text-destructive tabular-nums">
              {metrics?.rate_limited ?? "—"}
            </CardTitle>
          </CardHeader>
        </Card>
        <Card>
          <CardHeader>
            <CardDescription>Endpoints tracked</CardDescription>
            <CardTitle className="text-3xl text-foreground tabular-nums">
              {metrics ? Object.keys(metrics.endpoints).length : "—"}
            </CardTitle>
          </CardHeader>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Requests per endpoint</CardTitle>
          <CardDescription>From the hand-rolled JSON /metrics.</CardDescription>
        </CardHeader>
        <CardContent className="h-72">
          {!chartData ? (
            <Skeleton className="h-full w-full" />
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis
                  dataKey="name"
                  stroke="var(--foreground)"
                  fontSize={12}
                />
                <YAxis stroke="var(--foreground)" fontSize={12} allowDecimals={false} />
                <Tooltip
                  contentStyle={{
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    color: "var(--foreground)",
                  }}
                />
                <Bar
                  dataKey="count"
                  fill="var(--primary)"
                  radius={[4, 4, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Held-out top-5 accuracy</CardTitle>
          <CardDescription>
            Live from <code className="text-xs">/eval/summary</code> — all
            available models evaluated on the same deterministic 80/20
            split of the sample corpus. Higher is better.
          </CardDescription>
        </CardHeader>
        <CardContent className="h-72">
          {!evalRows ? (
            <Skeleton className="h-full w-full" />
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={evalRows.map((r) => ({
                  model: r.model,
                  "top-1": r.top1,
                  "top-5": r.top5,
                }))}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="model" stroke="var(--foreground)" fontSize={12} />
                <YAxis
                  stroke="var(--foreground)"
                  fontSize={12}
                  tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                  domain={[0, 1]}
                />
                <Tooltip
                  formatter={(v) =>
                    typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v)
                  }
                  contentStyle={{
                    background: "var(--card)",
                    border: "1px solid var(--border)",
                    borderRadius: 6,
                    color: "var(--foreground)",
                  }}
                />
                <Bar dataKey="top-1" fill="var(--primary)" radius={[4, 4, 0, 0]} />
                <Bar dataKey="top-5" fill="var(--accent)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="flex flex-row items-start justify-between">
          <div>
            <CardTitle>Prometheus exposition</CardTitle>
            <CardDescription>
              Raw output of{" "}
              <code className="text-xs">/metrics/prom</code> (first 40 lines).
            </CardDescription>
          </div>
          <Badge className="bg-accent text-accent-foreground hover:bg-accent">
            scrape me
          </Badge>
        </CardHeader>
        <CardContent>
          {prom === null ? (
            <Skeleton className="h-48 w-full" />
          ) : (
            <pre className="text-xs bg-muted rounded-md p-3 overflow-auto max-h-80 leading-relaxed">
              {prom.split("\n").slice(0, 40).join("\n")}
            </pre>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
