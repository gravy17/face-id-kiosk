// src/components/ui/index.tsx
import { clsx } from 'clsx'
import { Loader2 } from 'lucide-react'

// ── Card ───────────────────────────────────────────────────────────────────

export function Card({
  children,
  className,
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div
      className={clsx(
        'rounded-card border border-[hsl(var(--border))] bg-[hsl(var(--surface-1))] p-5',
        className,
      )}
    >
      {children}
    </div>
  )
}

// ── Badge ──────────────────────────────────────────────────────────────────

type BadgeVariant = 'accent' | 'success' | 'danger' | 'warning' | 'muted'

const badgeStyles: Record<BadgeVariant, string> = {
  accent:  'bg-[hsl(var(--accent-bg))]  text-[hsl(var(--accent-text))]',
  success: 'bg-[hsl(var(--success-bg))] text-[hsl(var(--success-text))]',
  danger:  'bg-[hsl(var(--danger-bg))]  text-[hsl(var(--danger-text))]',
  warning: 'bg-[hsl(var(--warning-bg))] text-[hsl(var(--warning-text))]',
  muted:   'bg-[hsl(var(--surface-2))]  text-[hsl(var(--text-muted))]',
}

export function Badge({
  children,
  variant = 'muted',
  className,
}: {
  children: React.ReactNode
  variant?: BadgeVariant
  className?: string
}) {
  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full px-2 py-0.5 text-[11px] font-medium',
        badgeStyles[variant],
        className,
      )}
    >
      {children}
    </span>
  )
}

// ── StatusBar ─────────────────────────────────────────────────────────────

type StatusVariant = 'ok' | 'warn' | 'error' | 'info'

const statusStyles: Record<StatusVariant, string> = {
  ok:    'bg-[hsl(var(--success-bg))] text-[hsl(var(--success-text))] border-[hsl(var(--success)/0.3)]',
  warn:  'bg-[hsl(var(--warning-bg))] text-[hsl(var(--warning-text))] border-[hsl(var(--warning)/0.3)]',
  error: 'bg-[hsl(var(--danger-bg))]  text-[hsl(var(--danger-text))]  border-[hsl(var(--danger)/0.3)]',
  info:  'bg-[hsl(var(--accent-bg))]  text-[hsl(var(--accent-text))]  border-[hsl(var(--accent)/0.3)]',
}

export function StatusBar({
  children,
  variant = 'info',
  className,
}: {
  children: React.ReactNode
  variant?: StatusVariant
  className?: string
}) {
  return (
    <div
      className={clsx(
        'flex items-center gap-2 rounded-lg border px-3 py-2 text-[12px]',
        statusStyles[variant],
        className,
      )}
    >
      {children}
    </div>
  )
}

// ── Spinner ───────────────────────────────────────────────────────────────

export function Spinner({ className }: { className?: string }) {
  return (
    <Loader2
      className={clsx('animate-spin text-[hsl(var(--accent-text))]', className)}
      size={16}
    />
  )
}

// ── Button ────────────────────────────────────────────────────────────────

type ButtonVariant = 'primary' | 'ghost'

const btnStyles: Record<ButtonVariant, string> = {
  primary:
    'bg-[hsl(var(--accent))] text-white hover:opacity-90 active:opacity-80',
  ghost:
    'bg-[hsl(var(--surface-2))] text-[hsl(var(--text-secondary))] border border-[hsl(var(--border-strong))] hover:bg-[hsl(var(--border))]',
}

export function Button({
  children,
  variant = 'primary',
  disabled,
  loading,
  onClick,
  type = 'button',
  className,
}: {
  children: React.ReactNode
  variant?: ButtonVariant
  disabled?: boolean
  loading?: boolean
  onClick?: () => void
  type?: 'button' | 'submit'
  className?: string
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      className={clsx(
        'flex w-full items-center justify-center gap-2 rounded-lg px-4 py-2.5',
        'text-[13px] font-medium transition-opacity',
        'disabled:cursor-not-allowed disabled:opacity-40',
        btnStyles[variant],
        className,
      )}
    >
      {loading && <Spinner />}
      {children}
    </button>
  )
}

// ── Input ─────────────────────────────────────────────────────────────────

export function Input({
  label,
  optional,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement> & {
  label: string
  optional?: boolean
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="flex items-center gap-1.5 text-[11px] uppercase tracking-wide text-[hsl(var(--text-muted))]">
        {label}
        {optional && (
          <span className="rounded border border-[hsl(var(--border))] bg-[hsl(var(--surface-2))] px-1.5 py-px text-[10px] normal-case tracking-normal">
            optional
          </span>
        )}
      </label>
      <input
        {...props}
        className={clsx(
          'w-full rounded-lg border border-[hsl(var(--border-strong))] bg-[hsl(var(--surface-2))]',
          'px-3 py-1.5 text-[13px] text-[hsl(var(--text-primary))]',
          'placeholder:text-[hsl(var(--text-muted))]',
          'outline-none focus:border-[hsl(var(--accent))] focus:ring-1 focus:ring-[hsl(var(--accent)/0.3)]',
          props.className,
        )}
      />
    </div>
  )
}

// ── DataRow ───────────────────────────────────────────────────────────────

export function DataRow({
  label,
  value,
  mono,
}: {
  label: React.ReactNode
  value: React.ReactNode
  mono?: boolean
}) {
  return (
    <div className="flex items-center justify-between border-b border-[hsl(var(--border))] py-2.5 last:border-none">
      <span className="flex items-center gap-1.5 text-[12px] text-[hsl(var(--text-secondary))]">
        {label}
      </span>
      <span
        className={clsx(
          'text-[13px] font-medium text-[hsl(var(--text-primary))]',
          mono && 'font-mono text-[11px] text-[hsl(var(--text-muted))]',
        )}
      >
        {value}
      </span>
    </div>
  )
}
