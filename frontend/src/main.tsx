import { StrictMode, Suspense, lazy } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter, Routes, Route } from "react-router-dom"
import App from "./App"
import Overview from "./pages/Overview"
import { Skeleton } from "./components/ui/skeleton"
import "./index.css"

// Overview is the default route so keep it eager. Autocomplete, Attention,
// and Metrics are lazy — Recharts (the chart library used on Metrics +
// Attention) is bulky, and there's no reason to ship it to users who
// land on / and never navigate elsewhere. React Router v7 declarative
// mode works transparently with React.lazy + Suspense.
const Autocomplete = lazy(() => import("./pages/Autocomplete"))
const Attention = lazy(() => import("./pages/Attention"))
const Metrics = lazy(() => import("./pages/Metrics"))

function LazyFallback() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-1/3" />
      <Skeleton className="h-48 w-full" />
      <Skeleton className="h-48 w-full" />
    </div>
  )
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<Overview />} />
          <Route
            path="autocomplete"
            element={
              <Suspense fallback={<LazyFallback />}>
                <Autocomplete />
              </Suspense>
            }
          />
          <Route
            path="attention"
            element={
              <Suspense fallback={<LazyFallback />}>
                <Attention />
              </Suspense>
            }
          />
          <Route
            path="metrics"
            element={
              <Suspense fallback={<LazyFallback />}>
                <Metrics />
              </Suspense>
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
