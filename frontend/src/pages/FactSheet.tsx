// src/pages/FactSheet.tsx
import { fetchFactSheet } from '@/api/client'
import { Card, DataRow, Spinner, StatusBar } from '@/components/ui'
import { CalendarDays, CircleAlert, Lock, Palette, PawPrint, Pizza, ShieldCheck } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import type { FactSheetResponse } from '@/types'

export function FactSheet() {
  const location = useLocation()
  const navigate = useNavigate()

  const [data,    setData]    = useState<FactSheetResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)
  const [ttl,     setTtl]     = useState<number>(15 * 60)  // seconds

  // Token passed from Verify page via router state or sessionStorage
  const token  = (location.state as any)?.token  ?? sessionStorage.getItem('fs_token')
  const userId = (location.state as any)?.userId ?? sessionStorage.getItem('fs_user_id')

  useEffect(() => {
    if (!token) return

    setLoading(true)
    setError(null)

    // We need to discover the userId — fetch with a sentinel if not known yet.
    // The backend returns 403 if the token userId doesn't match the path,
    // so we pass the sub claim extracted from the JWT payload.
    const sub = extractSubFromJwt(token)
    if (!sub) {
      setError('Invalid verification token.')
      setLoading(false)
      return
    }

    fetchFactSheet(sub, token)
      .then(setData)
      .catch(err => setError(err?.response?.data?.detail ?? 'Failed to load fact sheet.'))
      .finally(() => setLoading(false))
  }, [token])

  // Countdown TTL display
  useEffect(() => {
    if (!data) return
    const id = setInterval(() => setTtl(t => Math.max(0, t - 1)), 1000)
    return () => clearInterval(id)
  }, [data])

  if (!token) {
    return (
      <div className="flex h-full flex-col gap-5 p-6">
        <h1 className="text-[16px] font-medium text-[hsl(var(--text-primary))]">Fact sheet</h1>
        <StatusBar variant="warn">
          <CircleAlert size={13} />
          Complete facial verification first to access a fact sheet.
        </StatusBar>
        <button
          onClick={() => navigate('/verify')}
          className="text-[13px] text-[hsl(var(--accent-text))] underline underline-offset-2"
        >
          Go to Verify →
        </button>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center">
        <Spinner className="h-5 w-5" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-full flex-col gap-5 p-6">
        <h1 className="text-[16px] font-medium text-[hsl(var(--text-primary))]">Fact sheet</h1>
        <StatusBar variant="error">
          <CircleAlert size={13} /> {error}
        </StatusBar>
      </div>
    )
  }

  if (!data) return null

  const initials = `${data.firstName[0]}${data.lastName[0]}`.toUpperCase()
  const minutes  = Math.floor(ttl / 60)
  const seconds  = ttl % 60

  return (
    <div className="flex h-full flex-col gap-5 overflow-y-auto p-6">
      <div>
        <h1 className="text-[16px] font-medium text-[hsl(var(--text-primary))]">Fact sheet</h1>
        <p className="mt-1 text-[13px] text-[hsl(var(--text-muted))]">
          Accessible only after successful facial verification.
        </p>
      </div>

      <Card className="mx-auto w-full max-w-sm animate-fade-in">
        {/* Header */}
        <div className="mb-5 flex items-center gap-4">
          <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-[hsl(var(--accent-bg))] text-[16px] font-medium text-[hsl(var(--accent-text))]">
            {initials}
          </div>
          <div>
            <p className="text-[16px] font-medium text-[hsl(var(--text-primary))]">
              {data.firstName} {data.lastName}
            </p>
            <p className="mt-0.5 flex items-center gap-1 text-[12px] text-[hsl(var(--success-text))]">
              <ShieldCheck size={12} /> Verified
            </p>
          </div>
        </div>

        {/* Rows */}
        {data.favoriteFood && (
          <DataRow
            label={<><Pizza size={13} /> Favorite food</>}
            value={data.favoriteFood}
          />
        )}
        {data.favoriteColor && (
          <DataRow
            label={<><Palette size={13} /> Favorite color</>}
            value={data.favoriteColor}
          />
        )}
        {data.petName && (
          <DataRow
            label={<><PawPrint size={13} /> Pet name</>}
            value={data.petName}
          />
        )}
        <DataRow
          label={<><CalendarDays size={13} /> Registered</>}
          value={new Date(data.createdAt).toLocaleDateString('en-GB', {
            day: 'numeric', month: 'long', year: 'numeric',
          })}
        />

        {/* Token bar */}
        <div className="mt-4 flex items-center gap-2 rounded-lg border border-[hsl(var(--border))] bg-[hsl(var(--surface-2))] px-3 py-2">
          <Lock size={13} className="flex-shrink-0 text-[hsl(var(--accent-text))]" />
          <span className="flex-1 truncate font-mono text-[10px] text-[hsl(var(--text-muted))]">
            {token.slice(0, 40)}…
          </span>
          <span className="flex-shrink-0 text-[11px] text-[hsl(var(--text-muted))]">
            {minutes}:{String(seconds).padStart(2, '0')}
          </span>
        </div>
      </Card>
    </div>
  )
}

// ── Helpers ────────────────────────────────────────────────────────────────

function extractSubFromJwt(token: string): string | null {
  try {
    const payload = JSON.parse(atob(token.split('.')[1]))
    return payload.sub ?? null
  } catch {
    return null
  }
}
