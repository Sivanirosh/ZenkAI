// Capture room hook (PIVOT_ROADMAP §B.6/§B.18).
//
// Owns getUserMedia({ video }) for the back camera (when available),
// renders frames to a hidden canvas on shutter, and POSTs the JPEG
// to /api/v1/capture/image. The room never blocks: if the camera is
// denied we fall back to file-input.

'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import { captureImage, captureRecent } from '@/lib/api'
import type {
  CaptureRecentRow,
  CaptureResultPayload,
  CaptureSurfaceKind,
} from '@/lib/types'

export type CaptureStatus =
  | 'idle'
  | 'requesting'
  | 'streaming'
  | 'capturing'
  | 'processing'
  | 'error'

export interface UseCaptureOptions {
  sessionId: string
  surfaceKind?: CaptureSurfaceKind
  targetCompetencyId?: string | null
  targetCefr?: string | null
  note?: string | null
}

export interface UseCaptureResult {
  status: CaptureStatus
  videoRef: React.RefObject<HTMLVideoElement>
  canvasRef: React.RefObject<HTMLCanvasElement>
  recent: CaptureRecentRow[]
  lastResult: CaptureResultPayload | null
  error: string | null
  cameraAvailable: boolean
  startCamera: () => Promise<void>
  stopCamera: () => void
  shutter: (override?: Partial<UseCaptureOptions>) => Promise<CaptureResultPayload | null>
  uploadFile: (file: File, override?: Partial<UseCaptureOptions>) => Promise<CaptureResultPayload | null>
  refreshRecent: () => Promise<void>
  reset: () => void
}

export function useCapture(opts: UseCaptureOptions): UseCaptureResult {
  const [status, setStatus] = useState<CaptureStatus>('idle')
  const [recent, setRecent] = useState<CaptureRecentRow[]>([])
  const [lastResult, setLastResult] = useState<CaptureResultPayload | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [cameraAvailable, setCameraAvailable] = useState(true)

  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const stopCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    if (status === 'streaming') setStatus('idle')
  }, [status])

  const startCamera = useCallback(async () => {
    setError(null)
    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      setCameraAvailable(false)
      setStatus('error')
      setError('Kamera nicht verfügbar')
      return
    }
    setStatus('requesting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play().catch(() => undefined)
      }
      setStatus('streaming')
    } catch (err) {
      setCameraAvailable(false)
      setStatus('error')
      setError(err instanceof Error ? err.message : String(err))
    }
  }, [])

  const sendBlob = useCallback(
    async (
      blob: Blob,
      override?: Partial<UseCaptureOptions>,
    ): Promise<CaptureResultPayload | null> => {
      setStatus('processing')
      try {
        const result = await captureImage({
          image: blob,
          sessionId: override?.sessionId ?? opts.sessionId,
          surfaceKind: override?.surfaceKind ?? opts.surfaceKind ?? 'text',
          targetCompetencyId:
            override?.targetCompetencyId ?? opts.targetCompetencyId ?? null,
          targetCefr: override?.targetCefr ?? opts.targetCefr ?? null,
          note: override?.note ?? opts.note ?? null,
        })
        setLastResult(result)
        setStatus(streamRef.current ? 'streaming' : 'idle')
        return result
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
        setStatus('error')
        return null
      }
    },
    [opts.note, opts.sessionId, opts.surfaceKind, opts.targetCefr, opts.targetCompetencyId],
  )

  const shutter = useCallback(
    async (override?: Partial<UseCaptureOptions>) => {
      if (!videoRef.current || !canvasRef.current || !streamRef.current) {
        setError('Kamera ist nicht aktiv')
        setStatus('error')
        return null
      }
      setStatus('capturing')
      const video = videoRef.current
      const canvas = canvasRef.current
      const w = video.videoWidth || 1280
      const h = video.videoHeight || 720
      canvas.width = w
      canvas.height = h
      const ctx = canvas.getContext('2d')
      if (!ctx) {
        setError('Canvas nicht verfügbar')
        setStatus('error')
        return null
      }
      ctx.drawImage(video, 0, 0, w, h)
      const blob = await new Promise<Blob | null>((resolve) =>
        canvas.toBlob(resolve, 'image/jpeg', 0.85),
      )
      if (!blob) {
        setError('Aufnahme fehlgeschlagen')
        setStatus('error')
        return null
      }
      return sendBlob(blob, override)
    },
    [sendBlob],
  )

  const uploadFile = useCallback(
    async (file: File, override?: Partial<UseCaptureOptions>) => {
      return sendBlob(file, override)
    },
    [sendBlob],
  )

  const refreshRecent = useCallback(async () => {
    try {
      const r = await captureRecent(10)
      setRecent(r.captures)
    } catch {
      // ignore — recent panel just stays empty
    }
  }, [])

  const reset = useCallback(() => {
    setLastResult(null)
    setError(null)
    setStatus(streamRef.current ? 'streaming' : 'idle')
  }, [])

  useEffect(() => () => stopCamera(), [stopCamera])

  return {
    status,
    videoRef,
    canvasRef,
    recent,
    lastResult,
    error,
    cameraAvailable,
    startCamera,
    stopCamera,
    shutter,
    uploadFile,
    refreshRecent,
    reset,
  }
}
