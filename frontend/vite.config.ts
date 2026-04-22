import path from "node:path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

// Vite dev server config.
// - tailwindcss(): zero-config Tailwind v4 integration (reads src/index.css).
// - @/ alias: imports like "@/components/ui/button" resolve to src/.
// - /api proxy: fetch("/api/autocomplete") hits FastAPI on :8010 during dev.
//   The backend doesn't mount routes under /api, so the prefix is stripped.
export default defineConfig({
  plugins: [tailwindcss(), react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8010",
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/api/, ""),
      },
    },
  },
  build: {
    // Split the bundle so the main chunk isn't dominated by Recharts
    // (Metrics page + Attention page only). React core and router go
    // together; Radix primitives (shadcn's runtime) stay separate.
    rollupOptions: {
      output: {
        manualChunks: (id: string) => {
          if (id.includes("node_modules/recharts")) return "charts"
          if (
            id.includes("node_modules/react-router") ||
            id.includes("node_modules/react-dom") ||
            /node_modules\/react\//.test(id)
          )
            return "react"
          if (
            id.includes("node_modules/radix-ui") ||
            id.includes("node_modules/@radix-ui")
          )
            return "radix"
          return undefined
        },
      },
    },
  },
})
