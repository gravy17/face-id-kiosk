// src/pages/Verify.tsx
import { dataUrlToBlob, signChallenge } from '@/api/client'
import { CameraCapture } from '@/components/ui/CameraCapture'
import { Badge, Button, Card, DataRow, StatusBar } from '@/components/ui'
import { useChallenge, useVerify } from '@/hooks'
import { CircleAlert, Fingerprint } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'

export function Verify() {
  const challenge = useChallenge()
  const verify    = useVerify()
  const navigate  = useNavigate()

  const [captured,  setCaptured]  = useState(false)
  const [preview,   setPreview]   = useState<string | null>(null)
  const [imageBlob, setImageBlob] = useState<Blob | null>(null)
  const [captureTs, setCaptureTs] = useState<number>(0)

  function handleCapture(dataUrl: string, ts: number) {
    setPreview(dataUrl)
    setImageBlob(dataUrlToBlob(dataUrl))
    setCaptureTs(ts)
    setCaptured(true)
    verify.reset()
  }

  async function handleVerify() {
    if (!imageBlob || !challenge.data) return

    const { nonce } = challenge.data
    const sig        = await signChallenge(nonce, captureTs)

    verify.mutate(
      { image: imageBlob, nonce, capture_timestamp: captureTs, signature: sig },
      {
        onSuccess(data) {
          // Auto-navigate to fact sheet if verified + token present
          if (data.match.verified && data.token) {
            sessionStorage.setItem('fs_token',   data.token)
            sessionStorage.setItem('fs_user_id', '') // filled after factsheet fetch
            navigate('/factsheet', {
              state: { token: data.token, fromVerify: true },
            })
          }
        },
      },
    )
  }

  const result = verify.data

  return (
    <div className="flex h-full flex-col items-center gap-5 overflow-y-auto p-6">
      <div className="w-full">
        <h1 className="text-[16px] font-medium text-[hsl(var(--text-primary))]">Verify</h1>
        <p className="mt-1 text-[13px] text-[hsl(var(--text-muted))]">
          Match a live face against a registered identity.
        </p>
      </div>

      <div className="flex w-full max-w-sm flex-col gap-4">
        {/* Camera */}
        <CameraCapture
          onCapture={handleCapture}
          captured={captured}
          preview={preview}
          scanning
        />

        {/* Verify button */}
        <Button
          onClick={handleVerify}
          disabled={!captured}
          loading={verify.isPending}
        >
          <Fingerprint size={14} />
          Verify identity
        </Button>

        {/* Error */}
        {verify.isError && (
          <StatusBar variant="error" className="animate-fade-in">
            <CircleAlert size={13} />
            {(verify.error as any)?.response?.data?.detail ??
              'Verification failed — please try again.'}
          </StatusBar>
        )}

        {/* Result card */}
        {result && (
          <Card className="animate-fade-in">
            <p className="mb-3 text-[11px] uppercase tracking-wide text-[hsl(var(--text-muted))]">
              Result · {result.requestId}
            </p>

            <DataRow
              label="Liveness"
              value={
                <Badge variant={result.liveness.passed ? 'success' : 'danger'}>
                  {result.liveness.passed ? 'Passed' : 'Failed'}
                </Badge>
              }
            />
            <DataRow
              label="Liveness score"
              value={result.liveness.score?.toFixed(3) ?? '—'}
            />
            <DataRow
              label="Match"
              value={
                <Badge variant={result.match.verified ? 'success' : 'danger'}>
                  {result.match.verified ? 'Verified' : 'No match'}
                </Badge>
              }
            />
            {result.match.distance != null && (
              <DataRow
                label="Distance"
                value={`${result.match.distance.toFixed(3)} / ${result.match.threshold.toFixed(2)}`}
              />
            )}
            {result.match.confidence != null && (
              <DataRow
                label="Confidence"
                value={`${result.match.confidence.toFixed(1)}%`}
              />
            )}
            <DataRow
              label="Model"
              value={result.model.recognition}
              mono
            />
            <DataRow
              label="Time"
              value={`${result.performance.executionMs} ms`}
            />

            {result.match.verified && result.token && (
              <p className="mt-3 text-center text-[12px] text-[hsl(var(--success-text))]">
                Redirecting to fact sheet…
              </p>
            )}
          </Card>
        )}
      </div>
    </div>
  )
}
