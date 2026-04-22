import { NavLink, Outlet } from "react-router-dom"
import { useEffect, useState } from "react"
import { Moon, Sun, Activity, LayoutDashboard, Gauge, Type, Grid3x3 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Toaster } from "@/components/ui/sonner"
import { api } from "@/lib/api"
import { cn } from "@/lib/utils"

/** Sidebar nav item — active link picks up the primary color. */
function NavItem({
  to,
  icon: Icon,
  label,
}: {
  to: string
  icon: typeof LayoutDashboard
  label: string
}) {
  return (
    <NavLink
      to={to}
      end
      className={({ isActive }) =>
        cn(
          "flex items-center gap-3 rounded-md px-3 py-2 text-sm font-medium transition-colors",
          isActive
            ? "bg-primary text-primary-foreground"
            : "text-foreground hover:bg-muted",
        )
      }
    >
      <Icon className="size-4" />
      {label}
    </NavLink>
  )
}

/** Health pill polled on mount. Lime = healthy, red = down. */
function HealthPill() {
  const [state, setState] = useState<"loading" | "up" | "down">("loading")
  const [version, setVersion] = useState("")
  useEffect(() => {
    api
      .health()
      .then((r) => {
        setState("up")
        setVersion(r.version)
      })
      .catch(() => setState("down"))
  }, [])
  if (state === "loading") return <Badge variant="secondary">checking…</Badge>
  if (state === "down") return <Badge variant="destructive">API down</Badge>
  return (
    <Badge className="bg-accent text-accent-foreground hover:bg-accent">
      API {version}
    </Badge>
  )
}

/** Dark mode toggle — writes the `dark` class to <html>. */
function ThemeToggle() {
  const [dark, setDark] = useState(
    () => localStorage.getItem("theme") === "dark",
  )
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark)
    localStorage.setItem("theme", dark ? "dark" : "light")
  }, [dark])
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setDark((d) => !d)}
      aria-label="Toggle theme"
    >
      {dark ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </Button>
  )
}

export default function App() {
  return (
    <div className="min-h-screen flex">
      <aside className="w-60 shrink-0 border-r bg-card p-4 flex flex-col gap-4">
        <div className="flex items-center gap-2 px-1 pt-1">
          <Type className="size-5 text-primary" />
          <span className="font-semibold tracking-tight">Text Autocomplete</span>
        </div>
        <nav className="flex flex-col gap-1">
          <NavItem to="/" icon={LayoutDashboard} label="Overview" />
          <NavItem to="/autocomplete" icon={Activity} label="Autocomplete" />
          <NavItem to="/attention" icon={Grid3x3} label="Attention" />
          <NavItem to="/metrics" icon={Gauge} label="Metrics" />
        </nav>
        <div className="mt-auto flex items-center justify-between text-xs text-muted-foreground">
          <HealthPill />
          <ThemeToggle />
        </div>
      </aside>

      <main className="flex-1 p-8 max-w-5xl mx-auto w-full">
        <Outlet />
      </main>

      <Toaster position="top-right" richColors closeButton />
    </div>
  )
}
