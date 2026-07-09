// src/pages/Register.tsx
import { dataUrlToBlob, signChallenge } from '@/api/client'
import { CameraCapture } from '@/components/ui/CameraCapture'
import { Button, Card, Input, StatusBar } from '@/components/ui'
import { useChallenge, useRegister } from '@/hooks'
import { CheckCircle, CircleAlert, Info } from 'lucide-react'
import { useState } from 'react'

interface FormState {
  first_name:     string
  last_name:      string
  nickname:       string
  favorite_color: string
  favorite_food:  string
  pet_name:       string
}

const EMPTY: FormState = {
  first_name: '', last_name: '', nickname: '',
  favorite_color: '', favorite_food: '', pet_name: '',
}

export function Register() {
  const challenge = useChallenge()
  const register  = useRegister()

  const [form,      setForm]      = useState<FormState>(EMPTY)
  const [captured,  setCaptured]  = useState(false)
  const [preview,   setPreview]   = useState<string | null>(null)
  const [imageBlob, setImageBlob] = useState<Blob | null>(null)
  const [captureTs, setCaptureTs] = useState<number>(0)

  function field(name: keyof FormState) {
    return {
      value:    form[name],
      onChange: (e: React.ChangeEvent<HTMLInputElement>) =>
        setForm(f => ({ ...f, [name]: e.target.value })),
    }
  }

  function handleCapture(dataUrl: string, ts: number) {
    setPreview(dataUrl)
    setImageBlob(dataUrlToBlob(dataUrl))
    setCaptureTs(ts)
    setCaptured(true)
    register.reset()
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!imageBlob || !challenge.data) return

    const { nonce } = challenge.data
    const sig        = await signChallenge(nonce, captureTs)

    register.mutate({
      image: imageBlob,
      nonce,
      capture_timestamp: captureTs,
      signature: sig,
      ...form,
    })
  }

  const isReady = captured && form.first_name && form.last_name

  return (
    <div className="flex h-full flex-col gap-5 overflow-y-auto p-6">
      <div>
        <h1 className="text-[16px] font-medium text-[hsl(var(--text-primary))]">Register</h1>
        <p className="mt-1 text-[13px] text-[hsl(var(--text-muted))]">
          Capture a face and fill in user details to create a new identity.
        </p>
      </div>

      {/* Challenge status */}
      {challenge.isSuccess && (
        <StatusBar variant="ok">
          <Info size={13} /> Webcam challenge issued — ready to capture
        </StatusBar>
      )}
      {challenge.isError && (
        <StatusBar variant="error">
          <CircleAlert size={13} /> Could not reach server — check backend is running
        </StatusBar>
      )}

      <form onSubmit={handleSubmit} className="flex flex-col gap-5">

        {/* ── Camera (top) ─────────────────────────────────────── */}
        <CameraCapture
          onCapture={handleCapture}
          captured={captured}
          preview={preview}
        />

        {/* ── Form fields (below camera) ────────────────────────── */}
        <Card className="flex flex-col gap-3">
          <p className="text-[11px] uppercase tracking-wide text-[hsl(var(--text-muted))]">
            Identity details
          </p>

          <div className="grid grid-cols-2 gap-3">
            <Input label="First name" placeholder="John" required {...field('first_name')} />
            <Input label="Last name"  placeholder="Doe"  required {...field('last_name')} />
          </div>

          <Input label="Nickname" placeholder="JD" optional {...field('nickname')} />

          <div className="grid grid-cols-2 gap-3">
            <Input label="Favorite food"  placeholder="Jollof rice" optional {...field('favorite_food')} />
            <Input label="Favorite color" placeholder="Green"       optional {...field('favorite_color')} />
          </div>

          <Input label="Pet name" placeholder="Bruno" optional {...field('pet_name')} />
        </Card>

        {/* ── Submit ───────────────────────────────────────────── */}
        <Button
          type="submit"
          disabled={!isReady}
          loading={register.isPending}
        >
          Register identity
        </Button>

        {/* ── Result ───────────────────────────────────────────── */}
        {register.isSuccess && (
          <StatusBar variant="ok" className="animate-fade-in">
            <CheckCircle size={13} />
            Registered — ID{' '}
            <span className="font-mono text-[11px]">
              {register.data.userId.slice(0, 8)}…
            </span>
            {' '}· {register.data.processingTimeMs} ms
          </StatusBar>
        )}

        {register.isError && (
          <StatusBar variant="error" className="animate-fade-in">
            <CircleAlert size={13} />
            {(register.error as any)?.response?.data?.detail ??
              'Registration failed — please try again.'}
          </StatusBar>
        )}
      </form>
    </div>
  )
}
