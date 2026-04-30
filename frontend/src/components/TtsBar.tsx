// TTS playback bar for the Leseraum.
//
// Wraps a `useTts` hook and exposes play/pause, speed (0.8/1.0/1.2), a
// progress ticker ("Absatz 3/7"), and a backend hint (Piper vs. Browser).
// The parent Reader owns the chunk list so per-page text changes don't
// restart playback mid-sentence. The currently-playing id is surfaced via
// `onActiveChange` so paragraphs can paint a soft highlight.

'use client'

import { useEffect, useRef } from 'react'
import { useSession } from '@/store/session'
import { useTts, type TtsChunk } from '@/hooks/useTts'

export interface TtsBarProps {
  chunks: TtsChunk[]
  onActiveChange?: (id: string | null) => void
  onTabChange?: () => void
}

const SPEEDS: Array<0.8 | 1.0 | 1.2> = [0.8, 1.0, 1.2]

export function TtsBar({ chunks, onActiveChange, onTabChange }: TtsBarProps) {
  const readerPrefs = useSession((s) => s.readerPrefs)
  const setReaderPrefs = useSession((s) => s.setReaderPrefs)
  const setTab = useSession((s) => s.setTab)

  const tts = useTts(readerPrefs.ttsSpeed)

  useEffect(() => {
    tts.setSpeed(readerPrefs.ttsSpeed)
  }, [readerPrefs.ttsSpeed, tts])

  const lastActiveRef = useRef<string | null>(null)
  useEffect(() => {
    if (!onActiveChange) return
    if (tts.activeId === lastActiveRef.current) return
    lastActiveRef.current = tts.activeId
    onActiveChange(tts.activeId)
  }, [tts.activeId, onActiveChange])

  const isPlaying = tts.status === 'playing' || tts.status === 'loading'
  const isPaused = tts.status === 'paused'

  const handlePlayPause = () => {
    if (tts.status === 'idle' || tts.status === 'error') {
      tts.play(chunks, 0)
    } else if (isPlaying) {
      tts.pause()
    } else if (isPaused) {
      tts.resume()
    }
  }

  const total = Math.max(chunks.length, 1)
  const pos = Math.min(tts.currentIndex + (isPlaying ? 1 : 0), total)
  const progressPct = tts.status === 'idle' ? 0 : Math.round((pos / total) * 100)

  const backendLabel =
    tts.backend === 'piper'
      ? 'Piper \u00b7 de_DE-thorsten-high'
      : 'Browser TTS (Fallback)'

  return (
    <div className="tts-bar">
      <button
        className="tts-play"
        type="button"
        onClick={handlePlayPause}
        disabled={chunks.length === 0}
        title={isPlaying ? 'Pause' : 'Vorlesen'}
        aria-label={isPlaying ? 'Pause' : 'Vorlesen'}
      >
        {isPlaying ? '\u2759\u2759' : '\u25B6'}
      </button>
      <div className="tts-info">
        <div className="tts-label">
          {backendLabel}
          {tts.status !== 'idle' ? (
            <>
              {' \u00b7 '}
              Absatz {Math.min(tts.currentIndex + 1, total)}/{total}
            </>
          ) : null}
          {tts.error ? <> {'\u00b7'} {tts.error}</> : null}
        </div>
        <div className="tts-progress">
          <div
            className="tts-fill"
            style={{ width: `${progressPct}%` }}
          />
        </div>
      </div>
      <div className="speed-controls" role="group" aria-label="Vorlese-Tempo">
        {SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            className={s === readerPrefs.ttsSpeed ? 'speed-active' : 'speed-muted'}
            onClick={() => setReaderPrefs({ ttsSpeed: s })}
          >
            {s.toFixed(1)}
            {'\u00D7'}
          </button>
        ))}
      </div>
      <div className="divider-v" />
      <button
        className="mic-icon-btn"
        type="button"
        onClick={() => {
          tts.stop()
          if (onTabChange) onTabChange()
          else setTab('vx')
        }}
        aria-label="Stimme"
        title="Zum Stimme-Tab wechseln"
      >
        {'\uD83C\uDFA4'}
      </button>
    </div>
  )
}
