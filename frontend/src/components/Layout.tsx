import { NavLink, Outlet } from 'react-router-dom'
import { Pill, History } from 'lucide-react'

export default function Layout() {
  return (
    <div className="flex min-h-screen bg-pc-bg text-white">
      <aside className="w-56 border-r border-pc-border bg-pc-surface p-4">
        <h1 className="mb-8 text-lg font-bold">Fake Medicine Detection</h1>
        <nav className="flex flex-col gap-2">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `flex items-center gap-2 rounded-md px-3 py-2 ${isActive ? 'bg-pc-accent/20 text-pc-accent' : 'text-pc-muted'}`
            }
          >
            <Pill size={18} /> Check
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) =>
              `flex items-center gap-2 rounded-md px-3 py-2 ${isActive ? 'bg-pc-accent/20 text-pc-accent' : 'text-pc-muted'}`
            }
          >
            <History size={18} /> History
          </NavLink>
        </nav>
      </aside>
      <main className="flex-1 p-8">
        <Outlet />
      </main>
    </div>
  )
}
