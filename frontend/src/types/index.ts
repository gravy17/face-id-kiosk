// src/types/index.ts
// Mirror of backend Pydantic response schemas

export interface ChallengeResponse {
  nonce: string
  expires_at: number
}

export interface RegistrationResponse {
  requestId: string
  status: string
  userId: string
  embeddingModel: string
  detector: string
  processingTimeMs: number
}

export interface MatchDetail {
  verified: boolean
  distance: number | null
  threshold: number
  confidence: number | null
}

export interface LivenessDetail {
  passed: boolean
  score: number | null
}

export interface ModelDetail {
  recognition: string
  detector: string
}

export interface PerformanceDetail {
  executionMs: number
}

export interface VerificationResponse {
  requestId: string
  match: MatchDetail
  liveness: LivenessDetail
  model: ModelDetail
  performance: PerformanceDetail
  token: string | null
}

export interface FactSheetResponse {
  requestId: string
  userId: string
  firstName: string
  lastName: string
  nickname: string | null
  favoriteColor: string | null
  favoriteFood: string | null
  petName: string | null
  createdAt: string
  verified: boolean
}

export interface AuditLogEntry {
  id: string
  requestId: string
  action: string
  timestamp: string
  status: string
  clientIp: string
  executionMs: number | null
  details: Record<string, unknown> | null
}

export interface AuditLogResponse {
  requestId: string
  total: number
  limit: number
  offset: number
  logs: AuditLogEntry[]
}

export interface HealthResponse {
  status: string
  version: string
  environment: string
}

export interface ApiError {
  requestId: string
  status: 'error'
  detail: string
  code: string | null
}
