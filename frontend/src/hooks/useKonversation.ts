// Live voice loop hook (PIVOT_ROADMAP §B.5).
//
// Owns the MediaRecorder lifecycle for one Konversation turn. After
// recording stops we POST the audio blob to /api/v1/conversation/turn,
// receive { user_transcript, assistant_reply, ... }, append both to
// the in-memory log, and (best-effort) play the assistant reply via
// /api/v1/voice/tts so the room feels live.
//
// On a 503 from /voice/tts we silently fall back to the browser's
// SpeechSynthesis API so the demo still talks back even when Piper
// isn't installed locally.

'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import {
  ServiceUnavailableError,
  konversationRecent,
  konversationStream,
  scorePronunciation,
  ttsSynthesise,
} from '@/lib/api'
import type {
  KonversationTranscript,
  KonversationTurnResponse,
  PronunciationScore,
} from '@/lib/types'

export type KonversationStatus =
  | 'idle'
  | 'recording'
  | 'processing'
  | 'speaking'
  | 'error'

export interface UseKonversationOptions {
  sessionId: string
  competencyId?: string | null
  scenario?: string | null
  targetCefr?: string | null
}

export interface UseKonversationResult {
  status: KonversationStatus
  log: KonversationTranscript[]
  lastTurn: KonversationTurnResponse | null
  error: string | null
  toggle: () => Promise<void>
  start: () => Promise<void>
  stop: () => Promise<KonversationTurnResponse | null>
  refresh: () => Promise<void>
  clear: () => void
  ttsUnavailable: boolean
  score: (referenceText: string) => Promise<PronunciationScore | null>
  lastScore: PronunciationScore | null
  scoring: boolean
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

function speakViaBrowser(text: string): void {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return
  const u = new SpeechSynthesisUtterance(text)
  u.lang = 'de-DE'
  u.rate = 1.0
  window.speechSynthesis.cancel()
  window.speechSynthesis.speak(u)
}

export function useKonversation(
  opts: UseKonversationOptions,
): UseKonversationResult {
  const [status, setStatus] = useState<KonversationStatus>('idle')
  const [log, setLog] = useState<KonversationTranscript[]>([])
  const [lastTurn, setLastTurn] = useState<KonversationTurnResponse | null>(
    null,
  )
  const [error, setError] = useState<string | null>(null)
  const [ttsUnavailable, setTtsUnavailable] = useState(false)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const lastUserBlobRef = useRef<Blob | null>(null)
  const [lastScore, setLastScore] = useState<PronunciationScore | null>(null)
  const [scoring, setScoring] = useState(false)

  const cleanupStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    recorderRef.current = null
    chunksRef.current = []
  }, [])

  const playReply = useCallback(
    async (text: string): Promise<void> => {
      if (!text.trim()) return
      setStatus('speaking')
      try {
        const tts = await ttsSynthesise(text, 1.0)
        const audio = new Audio(tts.url)
        audioRef.current = audio
        await new Promise<void>((resolve) => {
          audio.onended = () => resolve()
          audio.onerror = () => resolve()
          audio.play().catch(() => resolve())
        })
        URL.revokeObjectURL(tts.url)
      } catch (err) {
        if (err instanceof ServiceUnavailableError) {
          setTtsUnavailable(true)
          speakViaBrowser(text)
        } else {
          speakViaBrowser(text)
        }
      } finally {
        setStatus('idle')
      }
    },
    [],
  )

  const start = useCallback(async () => {
    setError(null)
    if (typeof navigator === 'undefined' || !navigator.mediaDevices) {
      setStatus('error')
      setError('Mikrofon nicht unterstützt')
      return
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const mimeType = pickMimeType()
      const recorder = new MediaRecorder(
        stream,
        mimeType ? { mimeType } : undefined,
      )
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

  const stop = useCallback(async (): Promise<KonversationTurnResponse | null> => {
    const recorder = recorderRef.current
    if (!recorder || recorder.state === 'inactive') return null
    return new Promise<KonversationTurnResponse | null>((resolve) => {
      recorder.onstop = async () => {
        setStatus('processing')
        const mime = recorder.mimeType || 'audio/webm'
        const blob = new Blob(chunksRef.current, { type: mime })
        lastUserBlobRef.current = blob
        setLastScore(null)
        cleanupStream()

        // Streaming path: show user bubble as soon as STT finishes,
        // build assistant bubble token by token, play TTS on done.
        let assistantAccum = ''
        let resultTurn: KonversationTurnResponse | null = null

        try {
          await konversationStream(
            {
              audio: blob,
              sessionId: opts.sessionId,
              competencyId: opts.competencyId ?? null,
              scenario: opts.scenario ?? null,
              targetCefr: opts.targetCefr ?? null,
            },
            {
              onTranscribed: (userEntry) => {
                // Show 'Du' bubble immediately after STT — before LLM.
                setLog((prev) => [...prev, userEntry as KonversationTranscript])
              },
              onToken: (delta) => {
                assistantAccum += delta
                // Live-update the last (assistant) bubble while streaming.
                setLog((prev) => {
                  if (prev.length === 0) return prev
                  const last = prev[prev.length - 1]
                  if (last.role === 'assistant') {
                    return [
                      ...prev.slice(0, -1),
                      { ...last, text: assistantAccum },
                    ]
                  }
                  // First token — insert placeholder assistant entry.
                  return [
                    ...prev,
                    {
                      id: `atx-stream`,
                      session_id: opts.sessionId,
                      role: 'assistant',
                      text: assistantAccum,
                      occurred_at: new Date().toISOString(),
                    } as KonversationTranscript,
                  ]
                })
              },
              onDone: async (result) => {
                // Replace streaming placeholder with the final persisted entry.
                const turn = result as KonversationTurnResponse & {
                  duration_ms: number
                }
                resultTurn = turn
                setLastTurn(turn)
                setLog((prev) => {
                  const withoutPlaceholder = prev.filter(
                    (e) => e.id !== 'atx-stream',
                  )
                  return [...withoutPlaceholder, turn.assistant_reply as KonversationTranscript]
                })
                await playReply(turn.assistant_reply.text)
                resolve(turn)
              },
              onError: (err) => {
                setError(err instanceof Error ? err.message : String(err))
                setStatus('error')
                resolve(null)
              },
            },
          )
        } catch (err) {
          setError(err instanceof Error ? err.message : String(err))
          setStatus('error')
          resolve(null)
        }
      }
      recorder.stop()
    })
  }, [cleanupStream, opts.competencyId, opts.scenario, opts.sessionId, opts.targetCefr, playReply])

  const toggle = useCallback(async () => {
    if (status === 'recording') {
      await stop()
      return
    }
    if (status === 'speaking' && audioRef.current) {
      audioRef.current.pause()
      audioRef.current = null
      setStatus('idle')
      return
    }
    if (status === 'idle' || status === 'error') {
      await start()
    }
  }, [status, start, stop])

  const refresh = useCallback(async () => {
    try {
      const recent = await konversationRecent(20)
      const ordered = [...recent.transcripts].reverse()
      setLog(ordered)
    } catch {
      // soft-fail; keep whatever we already had in memory
    }
  }, [])

  const clear = useCallback(() => {
    setLog([])
    setLastTurn(null)
    setError(null)
    setLastScore(null)
    lastUserBlobRef.current = null
  }, [])

  const score = useCallback(
    async (referenceText: string): Promise<PronunciationScore | null> => {
      const blob = lastUserBlobRef.current
      if (!blob || !referenceText.trim()) return null
      setScoring(true)
      try {
        const result = await scorePronunciation({
          audio: blob,
          referenceText,
          targetCefr: opts.targetCefr ?? null,
        })
        setLastScore(result)
        return result
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err))
        return null
      } finally {
        setScoring(false)
      }
    },
    [opts.targetCefr],
  )

  useEffect(() => () => cleanupStream(), [cleanupStream])

  return {
    status,
    log,
    lastTurn,
    error,
    toggle,
    start,
    stop,
    refresh,
    clear,
    ttsUnavailable,
    score,
    lastScore,
    scoring,
  }
}
