// src/components/ui/CameraCapture.tsx
import { clsx } from 'clsx'
import { Camera } from 'lucide-react'
import { useCallback, useRef } from 'react'
import Webcam from 'react-webcam'
import { Button } from './index'

interface CameraCaptureProps {
  onCapture: (dataUrl: string, timestamp: number) => void
  onRetake: () => void
  captured: boolean
  preview?: string | null
  scanning?: boolean
  className?: string
}

export function CameraCapture({
  onCapture,
  onRetake,
  captured,
  preview,
  scanning = false,
  className,
}: CameraCaptureProps) {
  const webcamRef = useRef<Webcam>(null)

  // Capture and retake are distinct actions, not a toggle of the same
  // handler: once captured=true, <Webcam> unmounts (replaced by the <img>
  // preview) so webcamRef.current is null and getScreenshot() would
  // silently return undefined. Retake must reset state directly, never
  // go through getScreenshot().
  const handleCapture = useCallback(() => {
    const dataUrl = webcamRef.current?.getScreenshot()
    if (dataUrl) onCapture(dataUrl, Math.floor(Date.now() / 1000))
  }, [onCapture])

  const handleButtonClick = useCallback(() => {
    if (captured) onRetake()
    else handleCapture()
  }, [captured, onRetake, handleCapture])

  return (
    <div className={clsx('flex flex-col gap-3', className)}>
      {/* Camera frame */}
      <div className="relative overflow-hidden rounded-card border border-[hsl(var(--border))] bg-[hsl(var(--surface-1))]" style={{ aspectRatio: '4/3' }}>

        {/* Live feed or preview */}
        {captured && preview ? (
          <img src={preview} alt="Captured face" className="h-full w-full object-cover" />
        ) : (
          <Webcam
            ref={webcamRef}
            screenshotFormat="image/jpeg"
            screenshotQuality={0.92}
            videoConstraints={{ facingMode: 'user', width: 1280, height: 960 }}
            className="h-full w-full object-cover"
            mirrored
          />
        )}

        {/* Face guide oval */}
        <div
          className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2 rounded-[50%_50%_45%_45%] border-2 border-[hsl(var(--accent)/0.5)]"
          style={{ height: '70%', aspectRatio: '130 / 160' }}
        />

        {/* Corner brackets */}
        {(['tl', 'tr', 'bl', 'br'] as const).map(corner => (
          <CornerBracket key={corner} corner={corner} />
        ))}

        {/* Scan line — shown only during verify mode */}
        {scanning && !captured && (
          <div
            className="pointer-events-none absolute left-1/2 h-px w-32 -translate-x-1/2 bg-[hsl(var(--accent))]"
            style={{ animation: 'scan 2s ease-in-out infinite', top: '40%' }}
          />
        )}

        {/* Captured overlay */}
        {captured && (
          <div className="absolute inset-0 flex items-end justify-center bg-gradient-to-t from-black/40 to-transparent pb-3">
            <span className="rounded-full bg-black/50 px-3 py-1 text-[11px] text-white backdrop-blur-sm">
              Captured
            </span>
          </div>
        )}
      </div>

      {/* Hint */}
      {!captured && (
        <p className="text-center text-[12px] text-[hsl(var(--text-muted))]">
          Position your face within the guide
        </p>
      )}

      {/* Capture / Retake button */}
      <Button
        onClick={handleButtonClick}
        variant={captured ? 'ghost' : 'primary'}
      >
        <Camera size={14} />
        {captured ? 'Retake' : 'Capture'}
      </Button>
    </div>
  )
}

// ── Corner bracket helper ─────────────────────────────────────────────────

function CornerBracket({ corner }: { corner: 'tl' | 'tr' | 'bl' | 'br' }) {
  const base = 'pointer-events-none absolute h-4 w-4 border-[hsl(var(--accent-text))]'
  const styles = {
    tl: 'top-3 left-3 border-t-2 border-l-2 rounded-tl',
    tr: 'top-3 right-3 border-t-2 border-r-2 rounded-tr',
    bl: 'bottom-3 left-3 border-b-2 border-l-2 rounded-bl',
    br: 'bottom-3 right-3 border-b-2 border-r-2 rounded-br',
  }
  return <div className={clsx(base, styles[corner])} />
}
