// src/api/client.ts
import axios from 'axios'
import type {
  AuditLogResponse,
  ChallengeResponse,
  FactSheetResponse,
  RegistrationResponse,
  VerificationResponse,
} from '@/types'

const http = axios.create({
  baseURL: '/api',
  timeout: 30_000,
})

// ── Challenge ──────────────────────────────────────────────────────────────

export async function fetchChallenge(): Promise<ChallengeResponse> {
  const { data } = await http.post<ChallengeResponse>('/challenge')
  return data
}

// ── Registration ───────────────────────────────────────────────────────────

export interface RegisterPayload {
  image:            Blob
  nonce:            string
  capture_timestamp:number
  first_name:       string
  last_name:        string
  nickname?:        string
  favorite_color?:  string
  favorite_food?:   string
  pet_name?:        string
}

export async function registerUser(
  payload: RegisterPayload,
): Promise<RegistrationResponse> {
  const form = new FormData()
  form.append('image',             payload.image, 'capture.jpg')
  form.append('nonce',             payload.nonce)
  form.append('capture_timestamp', String(payload.capture_timestamp))
  form.append('first_name',        payload.first_name)
  form.append('last_name',         payload.last_name)
  if (payload.nickname)       form.append('nickname',       payload.nickname)
  if (payload.favorite_color) form.append('favorite_color', payload.favorite_color)
  if (payload.favorite_food)  form.append('favorite_food',  payload.favorite_food)
  if (payload.pet_name)       form.append('pet_name',       payload.pet_name)

  const { data } = await http.post<RegistrationResponse>('/register', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

// ── Verification ───────────────────────────────────────────────────────────

export interface VerifyPayload {
  image:             Blob
  nonce:             string
  capture_timestamp: number
}

export async function verifyUser(
  payload: VerifyPayload,
): Promise<VerificationResponse> {
  const form = new FormData()
  form.append('image',             payload.image, 'capture.jpg')
  form.append('nonce',             payload.nonce)
  form.append('capture_timestamp', String(payload.capture_timestamp))

  const { data } = await http.post<VerificationResponse>('/verify', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

// ── Fact Sheet ─────────────────────────────────────────────────────────────

export async function fetchFactSheet(
  userId: string,
  token:  string,
): Promise<FactSheetResponse> {
  const { data } = await http.get<FactSheetResponse>(`/factsheet/${userId}`, {
    headers: { Authorization: `Bearer ${token}` },
  })
  return data
}

// ── Audit Logs ─────────────────────────────────────────────────────────────

export async function fetchLogs(
  adminKey: string,
  limit  = 100,
  offset = 0,
): Promise<AuditLogResponse> {
  const { data } = await http.get<AuditLogResponse>('/logs', {
    headers: { Authorization: `Bearer ${adminKey}` },
    params:  { limit, offset },
  })
  return data
}

// ── Helpers ────────────────────────────────────────────────────────────────

/**
 * Convert a base64 data URL (from react-webcam) to a Blob.
 */
export function dataUrlToBlob(dataUrl: string): Blob {
  const [header, base64] = dataUrl.split(',')
  const mime             = header.match(/:(.*?);/)?.[1] ?? 'image/jpeg'
  const binary           = atob(base64)
  const arr              = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) arr[i] = binary.charCodeAt(i)
  return new Blob([arr], { type: mime })
}
