// src/components/layout/Sidebar.tsx
import { clsx } from 'clsx'
import { ClipboardList, Fingerprint, IdCard, UserPlus } from 'lucide-react'
import { NavLink } from 'react-router-dom'

const links = [
  { to: '/register',  label: 'Register',   Icon: UserPlus      },
  { to: '/verify',    label: 'Verify',      Icon: Fingerprint   },
  { to: '/factsheet', label: 'Fact sheet',  Icon: IdCard        },
  { to: '/logs',      label: 'Audit logs',  Icon: ClipboardList },
]

export function Sidebar() {
  return (
    <aside className="flex w-44 flex-shrink-0 flex-col border-r border-[hsl(var(--border))] bg-[hsl(var(--surface-1))]">
      {/* Logo */}
      <div className="border-b border-[hsl(var(--border))] px-4 py-5">
        <p className="text-[13px] font-medium text-[hsl(var(--text-primary))]">
          FaceID Kiosk
        </p>
        <p className="mt-0.5 text-[11px] text-[hsl(var(--text-muted))]">
          Identity management
        </p>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-0.5 p-2 pt-3">
        {links.map(({ to, label, Icon }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-2.5 rounded-lg px-3 py-2 text-[13px] transition-colors',
                isActive
                  ? 'bg-[hsl(var(--accent-bg))] font-medium text-[hsl(var(--accent-text))]'
                  : 'text-[hsl(var(--text-secondary))] hover:bg-[hsl(var(--surface-2))]',
              )
            }
          >
            <Icon size={15} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Status dot */}
      <div className="mt-auto flex items-center gap-2 px-4 py-4 text-[11px] text-[hsl(var(--text-muted))]">
        <span className="h-1.5 w-1.5 rounded-full bg-[hsl(var(--success))]" />
        Connected
      </div>
    </aside>
  )
}
