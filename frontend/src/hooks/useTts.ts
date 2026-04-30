// Paragraph-sequenced TTS for the Leseraum and Voice answers.
//
// The hook accepts a list of `{id, text}` chunks (paragraphs, sentences, or a
// single answer blob) and plays them back in order. Each chunk is fetched
// from /voice/tts as a WAV blob, decoded by the browser's <audio> element,
// then played sequentially. The `currentIndex` / `activeId` outputs let the
// Reader paint a soft highlight on the chunk currently being read.
//
// When Piper isn't installed, /voice/tts returns 503; we fall back to the
// browser's built-in SpeechSynthesis so the loop never breaks.

'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { ServiceUnavailableError, ttsSynthesise } from '@/lib/api'

export interface TtsChunk {
  id: string
  text: string
}

export type TtsStatus = 'idle' | 'loading' | 'playing' | 'paused' | 'error'

export interface UseTtsResult {
  status: TtsStatus
  activeId: string | null
  currentIndex: number
  chunkCount: number
  backend: 'piper' | 'browser'
  error: string | null
  play: (chunks: TtsChunk[], startIndex?: number) => void
  pause: () => void
  resume: () => void
  stop: () => void
  skipTo: (idx: number) => void
  next: () => void
  prev: () => void
  setSpeed: (speed: number) => void
  speed: number
}

export function useTts(initialSpeed = 1.0): UseTtsResult {
  const [status, setStatus] = useState<TtsStatus>('idle')
  const [currentIndex, setCurrentIndex] = useState(0)
  const [chunks, setChunks] = useState<TtsChunk[]>([])
  const [backend, setBackend] = useState<'piper' | 'browser'>('piper')
  const [error, setError] = useState<string | null>(null)
  const [speed, setSpeed] = useState(initialSpeed)

  const audioRef = useRef<HTMLAudioElement | null>(null)
  const objectUrlRef = useRef<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const browserUtteranceRef = useRef<SpeechSynthesisUtterance | null>(null)
  const runIdRef = useRef(0)

  const teardownAudio = useCallback(() => {
    if (audioRef.current) {
      audioRef.current.pause()
      audioRef.current.src = ''
      audioRef.current = null
    }
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current)
      objectUrlRef.current = null
    }
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    if (browserUtteranceRef.current && typeof window !== 'undefined') {
      window.speechSynthesis.cancel()
      browserUtteranceRef.current = null
    }
  }, [])

  const stop = useCallback(() => {
    runIdRef.current += 1
    teardownAudio()
    setStatus('idle')
    setCurrentIndex(0)
  }, [teardownAudio])

  const playBrowser = useCallback(
    (list: TtsChunk[], startIndex: number, localRunId: number) => {
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
        setStatus('error')
        setError('Keine TTS verfügbar (weder Piper noch Browser)')
        return
      }
      setBackend('browser')
      const voices = window.speechSynthesis.getVoices()
      const germanVoice =
        voices.find((v) => v.lang.startsWith('de')) ??
        voices.find((v) => v.default) ??
        voices[0] ??
        null

      const speak = (idx: number) => {
        if (runIdRef.current !== localRunId) return
        if (idx >= list.length) {
          setStatus('idle')
          setCurrentIndex(list.length)
          return
        }
        setCurrentIndex(idx)
        setStatus('playing')
        const utter = new SpeechSynthesisUtterance(list[idx].text)
        utter.lang = germanVoice?.lang ?? 'de-DE'
        utter.rate = speed
        if (germanVoice) utter.voice = germanVoice
        utter.onend = () => {
          if (runIdRef.current !== localRunId) return
          speak(idx + 1)
        }
        utter.onerror = () => {
          if (runIdRef.current !== localRunId) return
          setStatus('error')
          setError('Browser-TTS fehlgeschlagen')
        }
        browserUtteranceRef.current = utter
        window.speechSynthesis.speak(utter)
      }
      speak(startIndex)
    },
    [speed],
  )

  const playPiper = useCallback(
    async (list: TtsChunk[], startIndex: number, localRunId: number) => {
      setBackend('piper')
      for (let i = startIndex; i < list.length; i++) {
        if (runIdRef.current !== localRunId) return
        const chunk = list[i]
        setCurrentIndex(i)
        setStatus('loading')
        try {
          const { url } = await ttsSynthesise(chunk.text, speed)
          if (runIdRef.current !== localRunId) {
            URL.revokeObjectURL(url)
            return
          }
          if (objectUrlRef.current) {
            URL.revokeObjectURL(objectUrlRef.current)
          }
          objectUrlRef.current = url

          const audio = new Audio(url)
          audio.playbackRate = 1.0 // speed is baked into Piper length_scale
          audioRef.current = audio
          setStatus('playing')

          await new Promise<void>((resolve, reject) => {
            audio.onended = () => resolve()
            audio.onerror = () =>
              reject(new Error('audio decode error'))
            audio.play().catch(reject)
          })
        } catch (err) {
          if (runIdRef.current !== localRunId) return
          if (err instanceof ServiceUnavailableError) {
            // Restart from this chunk using browser TTS.
            playBrowser(list, i, localRunId)
            return
          }
          if (err instanceof Error && err.name === 'AbortError') return
          setStatus('error')
          setError(err instanceof Error ? err.message : String(err))
          return
        }
      }
      if (runIdRef.current === localRunId) {
        setStatus('idle')
        setCurrentIndex(list.length)
      }
    },
    [speed, playBrowser],
  )

  const play = useCallback(
    (list: TtsChunk[], startIndex = 0) => {
      teardownAudio()
      const localRunId = ++runIdRef.current
      setError(null)
      setChunks(list)
      if (list.length === 0) {
        setStatus('idle')
        return
      }
      void playPiper(list, startIndex, localRunId)
    },
    [playPiper, teardownAudio],
  )

  const pause = useCallback(() => {
    if (backend === 'piper' && audioRef.current) {
      audioRef.current.pause()
      setStatus('paused')
    } else if (backend === 'browser' && typeof window !== 'undefined') {
      window.speechSynthesis.pause()
      setStatus('paused')
    }
  }, [backend])

  const resume = useCallback(() => {
    if (backend === 'piper' && audioRef.current) {
      audioRef.current.play().catch(() => undefined)
      setStatus('playing')
    } else if (backend === 'browser' && typeof window !== 'undefined') {
      window.speechSynthesis.resume()
      setStatus('playing')
    }
  }, [backend])

  const skipTo = useCallback(
    (idx: number) => {
      if (chunks.length === 0) return
      play(chunks, Math.max(0, Math.min(idx, chunks.length - 1)))
    },
    [chunks, play],
  )

  const next = useCallback(() => skipTo(currentIndex + 1), [skipTo, currentIndex])
  const prev = useCallback(() => skipTo(currentIndex - 1), [skipTo, currentIndex])

  useEffect(() => () => teardownAudio(), [teardownAudio])

  const activeId =
    status === 'playing' || status === 'paused' || status === 'loading'
      ? chunks[currentIndex]?.id ?? null
      : null

  return {
    status,
    activeId,
    currentIndex,
    chunkCount: chunks.length,
    backend,
    error,
    play,
    pause,
    resume,
    stop,
    skipTo,
    next,
    prev,
    setSpeed,
    speed,
  }
}
