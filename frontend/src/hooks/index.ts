// src/hooks/index.ts
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  fetchChallenge,
  fetchFactSheet,
  fetchLogs,
  registerUser,
  verifyUser,
  type RegisterPayload,
  type VerifyPayload,
} from '@/api/client'

// ── Challenge ──────────────────────────────────────────────────────────────

export function useChallenge() {
  return useQuery({
    queryKey: ['challenge'],
    queryFn:  fetchChallenge,
    staleTime: 20_000,   // refresh before 30s TTL expires
    refetchInterval: 20_000,
  })
}

// ── Registration ───────────────────────────────────────────────────────────

export function useRegister() {
  return useMutation({
    mutationFn: (payload: RegisterPayload) => registerUser(payload),
  })
}

// ── Verification ───────────────────────────────────────────────────────────

export function useVerify() {
  return useMutation({
    mutationFn: (payload: VerifyPayload) => verifyUser(payload),
  })
}

// ── Fact Sheet ─────────────────────────────────────────────────────────────

export function useFactSheet(userId: string, token: string, enabled: boolean) {
  return useQuery({
    queryKey: ['factsheet', userId],
    queryFn:  () => fetchFactSheet(userId, token),
    enabled,
    staleTime: 10 * 60 * 1000,  // 10 min — within JWT TTL
  })
}

// ── Audit Logs ─────────────────────────────────────────────────────────────

export function useLogs(adminKey: string, limit = 100, offset = 0) {
  return useQuery({
    queryKey: ['logs', adminKey, limit, offset],
    queryFn:  () => fetchLogs(adminKey, limit, offset),
    enabled:  adminKey.length > 0,
    refetchInterval: 10_000,
  })
}
