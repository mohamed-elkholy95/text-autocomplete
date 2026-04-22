import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter, Routes, Route } from "react-router-dom"
import App from "./App"
import Overview from "./pages/Overview"
import Autocomplete from "./pages/Autocomplete"
import Metrics from "./pages/Metrics"
import "./index.css"

// React Router v7 in declarative mode: App is the layout (sidebar + <Outlet/>),
// nested routes render into the main pane.
createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<Overview />} />
          <Route path="autocomplete" element={<Autocomplete />} />
          <Route path="metrics" element={<Metrics />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
