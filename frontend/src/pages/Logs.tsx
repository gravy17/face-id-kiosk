// src/pages/Logs.tsx
import { Badge, Card, DataRow, Spinner, StatusBar } from '@/components/ui'
import { useLogs } from '@/hooks'
import type { AuditLogEntry } from '@/types'
import { CircleAlert, RefreshCw, Shield } from 'lucide-react'
import { useState } from 'react'

const ACTION_VARIANTS: Record<string, 'accent' | 'success' | 'warning' | 'muted'> = {
  REGISTER: 'accent',
  VERIFY:   'success',
}

const STATUS_VARIANTS: Record<string, 'success' | 'danger' | 'warning' | 'muted'> = {
  SUCCESS:             'success',
  VERIFIED:            'success',
  REJECTED_LIVENESS:   'danger',
  REJECTED_DUPLICATE:  'danger',
  LIVENESS_FAIL:       'danger',
  NO_MATCH:            'warning',
}

export function Logs() {
  const [adminKey,      setAdminKey]      = useState('')
  const [submittedKey,  setSubmittedKey]  = useState('')
  const [actionFilter,  setActionFilter]  = useState('')
  const [statusFilter,  setStatusFilter]  = useState('')

  const { data, isLoading, isError, refetch } = useLogs(submittedKey)

  const logs = (data?.logs ?? []).filter(l => {
    if (actionFilter && l.action !== actionFilter) return false
    if (statusFilter && l.status !== statusFilter) return false
    return true
  })

  return (
    <div className="flex h-full flex-col gap-5 p-6">
      <div>
        <h1 className="text-[16px] font-medium text-[hsl(var(--text-primary))]">Audit logs</h1>
        <p className="mt-1 text-[13px] text-[hsl(var(--text-muted))]">
          Admin access only. All registration and verification events.
        </p>
      </div>

      {/* Admin key input */}
      {!submittedKey && (
        <Card className="flex flex-col gap-3">
          <div className="flex items-center gap-2 text-[13px] text-[hsl(var(--text-secondary))]">
            <Shield size={14} className="text-[hsl(var(--accent-text))]" />
            Enter your admin API key to view logs
          </div>
          <div className="flex gap-2">
            <input
              type="password"
              value={adminKey}
              onChange={e => setAdminKey(e.target.value)}
              placeholder="Admin key…"
              className="flex-1 rounded-lg border border-[hsl(var(--border-strong))] bg-[hsl(var(--surface-2))] px-3 py-1.5 text-[13px] text-[hsl(var(--text-primary))] outline-none focus:border-[hsl(var(--accent))]"
            />
            <button
              onClick={() => setSubmittedKey(adminKey)}
              disabled={!adminKey}
              className="rounded-lg bg-[hsl(var(--accent))] px-4 py-1.5 text-[13px] font-medium text-white disabled:opacity-40"
            >
              View
            </button>
          </div>
        </Card>
      )}

      {isError && (
        <StatusBar variant="error">
          <CircleAlert size={13} />
          Invalid admin key or server error.{' '}
          <button
            onClick={() => setSubmittedKey('')}
            className="underline underline-offset-2"
          >
            Try again
          </button>
        </StatusBar>
      )}

      {isLoading && (
        <div className="flex items-center gap-2 text-[13px] text-[hsl(var(--text-muted))]">
          <Spinner /> Loading logs…
        </div>
      )}

      {data && (
        <>
          {/* Filters + refresh */}
          <div className="flex items-center gap-2">
            <select
              value={actionFilter}
              onChange={e => setActionFilter(e.target.value)}
              className="rounded-lg border border-[hsl(var(--border-strong))] bg-[hsl(var(--surface-2))] px-2 py-1.5 text-[12px] text-[hsl(var(--text-primary))] outline-none"
            >
              <option value="">All actions</option>
              <option value="REGISTER">REGISTER</option>
              <option value="VERIFY">VERIFY</option>
            </select>
            <select
              value={statusFilter}
              onChange={e => setStatusFilter(e.target.value)}
              className="rounded-lg border border-[hsl(var(--border-strong))] bg-[hsl(var(--surface-2))] px-2 py-1.5 text-[12px] text-[hsl(var(--text-primary))] outline-none"
            >
              <option value="">All statuses</option>
              <option value="SUCCESS">SUCCESS</option>
              <option value="VERIFIED">VERIFIED</option>
              <option value="NO_MATCH">NO_MATCH</option>
              <option value="LIVENESS_FAIL">LIVENESS_FAIL</option>
              <option value="REJECTED_DUPLICATE">REJECTED_DUPLICATE</option>
            </select>
            <button
              onClick={() => refetch()}
              className="ml-auto flex items-center gap-1.5 text-[12px] text-[hsl(var(--text-muted))] hover:text-[hsl(var(--text-secondary))]"
            >
              <RefreshCw size={12} /> Refresh
            </button>
          </div>

          {/* Table */}
          <Card className="overflow-x-auto p-0">
            <table className="w-full text-[12px]">
              <thead>
                <tr className="border-b border-[hsl(var(--border))]">
                  {['Time', 'Action', 'Request ID', 'Status', 'Score', 'ms'].map(h => (
                    <th
                      key={h}
                      className="px-4 py-2.5 text-left text-[11px] font-medium uppercase tracking-wide text-[hsl(var(--text-muted))]"
                    >
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {logs.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-6 text-center text-[hsl(var(--text-muted))]">
                      No logs match your filters.
                    </td>
                  </tr>
                ) : (
                  logs.map(log => <LogRow key={log.id} log={log} />)
                )}
              </tbody>
            </table>
          </Card>

          <p className="text-right text-[11px] text-[hsl(var(--text-muted))]">
            {logs.length} of {data.total} entries
          </p>
        </>
      )}
    </div>
  )
}

// ── Log row ───────────────────────────────────────────────────────────────

function LogRow({ log }: { log: AuditLogEntry }) {
  const details   = log.details as any
  const score     = details?.match?.distance ?? details?.liveness?.score ?? null
  const timeStr   = new Date(log.timestamp).toLocaleTimeString('en-GB', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })

  return (
    <tr className="border-b border-[hsl(var(--border))] last:border-none hover:bg-[hsl(var(--surface-2)/0.5)]">
      <td className="px-4 py-2.5 font-mono text-[11px] text-[hsl(var(--text-muted))]">
        {timeStr}
      </td>
      <td className="px-4 py-2.5">
        <Badge variant={ACTION_VARIANTS[log.action] ?? 'muted'}>
          {log.action}
        </Badge>
      </td>
      <td className="px-4 py-2.5 font-mono text-[11px] text-[hsl(var(--text-muted))]">
        {log.requestId}
      </td>
      <td className="px-4 py-2.5">
        <span className="flex items-center gap-1.5 text-[hsl(var(--text-secondary))]">
          <span
            className={[
              'h-1.5 w-1.5 rounded-full flex-shrink-0',
              STATUS_VARIANTS[log.status] === 'success' ? 'bg-[hsl(var(--success))]'
              : STATUS_VARIANTS[log.status] === 'danger'  ? 'bg-[hsl(var(--danger))]'
              : STATUS_VARIANTS[log.status] === 'warning' ? 'bg-[hsl(var(--warning))]'
              : 'bg-[hsl(var(--border-strong))]',
            ].join(' ')}
          />
          {log.status}
        </span>
      </td>
      <td className="px-4 py-2.5 text-[hsl(var(--text-secondary))]">
        {score != null ? score.toFixed(3) : '—'}
      </td>
      <td className="px-4 py-2.5 text-[hsl(var(--text-secondary))]">
        {log.executionMs ?? '—'}
      </td>
    </tr>
  )
}
