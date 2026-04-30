// MediaRecorder → backend /voice/stt → transcript.
//
// The hook owns the recorder lifecycle and flips between `idle`, `recording`,
// and `processing` states. On a 503 from the backend (faster-whisper not
// installed) it surfaces `unavailable` so the mic button can show a tooltip
// instead of silently failing.

'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { ServiceUnavailableError, sttTranscribe } from '@/lib/api'

export type SttStatus = 'idle' | 'recording' | 'processing' | 'error'

export interface UseSttResult {
  status: SttStatus
  transcript: string | null
  error: string | null
  unavailable: boolean
  installHint: string | null
  start: () => Promise<void>
  stop: () => Promise<string | null>
  toggle: () => Promise<string | null>
  reset: () => void
}

function pickMimeType(): string {
  if (typeof window === 'undefined' || !('MediaRecorder' in window)) return ''
  const candidates = [
    'audio/webm;codecs=opus',
    'audio/webm',
    'audio/ogg;codecs=opus',
    'audio/mp4',
    'audio/wav',
  ]
  for (const mt of candidates) {
    if ((window as any).MediaRecorder.isTypeSupported?.(mt)) return mt
  }
  return ''
}

export function useStt(): UseSttResult {
  const [status, setStatus] = useState<SttStatus>('idle')
  const [transcript, setTranscript] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [unavailable, setUnavailable] = useState(false)
  const [installHint, setInstallHint] = useState<string | null>(null)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const streamRef = useRef<MediaStream | null>(null)

  const cleanupStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    recorderRef.current = null
    chunksRef.current = []
  }, [])

  const start = useCallback(async () => {
    setError(null)
    setTranscript(null)
    if (typeof navigator === 'undefined' || !navigator.mediaDevices) {
      setStatus('error')
      setError('Mikrofon nicht unterstützt')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const mimeType = pickMimeType()
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      chunksRef.current = []
      recorder.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) chunksRef.current.push(ev.data)
      }
      recorderRef.current = recorder
      recorder.start()
      setStatus('recording')
    } catch (err) {
      setStatus('error')
      setError(err instanceof Error ? err.message : String(err))
      cleanupStream()
    }
  }, [cleanupStream])

  const stop = useCallback(async (): Promise<string | null> => {
    const recorder = recorderRef.current
    if (!recorder || recorder.state === 'inactive') return null
    return new Promise<string | null>((resolve) => {
      recorder.onstop = async () => {
        setStatus('processing')
        const mime = recorder.mimeType || 'audio/webm'
        const blob = new Blob(chunksRef.current, { type: mime })
        cleanupStream()
        try {
          const result = await sttTranscribe(blob)
          setTranscript(result.transcript)
          setStatus('idle')
          resolve(result.transcript)
        } catch (err) {
          if (err instanceof ServiceUnavailableError) {
            setUnavailable(true)
            setInstallHint(err.info.install_hint || null)
            setError(err.info.detail)
          } else {
            setError(err instanceof Error ? err.message : String(err))
          }
          setStatus('error')
          resolve(null)
        }
      }
      recorder.stop()
    })
  }, [cleanupStream])

  const toggle = useCallback(async () => {
    if (status === 'recording') return stop()
    await start()
    return null
  }, [status, start, stop])

  const reset = useCallback(() => {
    setTranscript(null)
    setError(null)
    setStatus('idle')
  }, [])

  useEffect(() => () => cleanupStream(), [cleanupStream])

  return {
    status,
    transcript,
    error,
    unavailable,
    installHint,
    start,
    stop,
    toggle,
    reset,
  }
}
