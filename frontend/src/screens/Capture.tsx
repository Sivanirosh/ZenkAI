// Capture room (PIVOT_ROADMAP §B.6/§B.18).
//
// Live camera preview, big shutter, and a transcript panel that reveals
// itself when Mira finishes describing the frame. Falls back to file
// upload if the camera is denied or unavailable.

'use client'

import { useEffect, useMemo, useRef, useState } from 'react'

import { useCapture } from '@/hooks/useCapture'
import type { CaptureSurfaceKind } from '@/lib/types'

const SURFACES: { id: CaptureSurfaceKind; label: string; sub: string }[] = [
  { id: 'text', label: 'Text', sub: 'Schild, Buch, Zettel' },
  { id: 'menu', label: 'Speisekarte', sub: 'Café, Restaurant' },
  { id: 'sign', label: 'Schild', sub: 'Wegweiser, Plakat' },
  { id: 'object', label: 'Gegenstand', sub: 'Was ist das auf Deutsch?' },
]

function makeSessionId(): string {
  if (typeof window !== 'undefined' && window.crypto?.randomUUID) {
    return `cap-${window.crypto.randomUUID().slice(0, 12)}`
  }
  return `cap-${Math.random().toString(36).slice(2, 14)}`
}

export function Capture(): JSX.Element {
  const sessionRef = useRef<string>(makeSessionId())
  const [surfaceKind, setSurfaceKind] = useState<CaptureSurfaceKind>('text')
  const [note, setNote] = useState<string>('')

  const c = useCapture({
    sessionId: sessionRef.current,
    surfaceKind,
    note,
  })

  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    void c.refreshRecent()
    // single shot at mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const stageMessage = useMemo(() => {
    switch (c.status) {
      case 'requesting':
        return 'Kamera wird angefragt …'
      case 'streaming':
        return 'Kamera bereit. Visier dein Motiv und tippe Aufnahme.'
      case 'capturing':
        return 'Bild aufgenommen …'
      case 'processing':
        return 'Mira liest das Bild …'
      case 'error':
        return c.error || 'Fehler.'
      default:
        return 'Tippe auf Kamera starten oder lade ein Bild hoch.'
    }
  }, [c.status, c.error])

  return (
    <div className="capture">
      <header className="capture-header">
        <p className="capture-eyebrow">Capture</p>
        <h1 className="capture-title">Die Welt als Lehrbuch</h1>
        <p className="capture-sub">
          Ein Foto vom Schild, von der Speisekarte, von dem, was du gerade
          gesehen hast — Mira liest es, übersetzt, und schlägt drei Wörter zum
          Mitnehmen vor.
        </p>
      </header>

      <div className="capture-controls" role="group" aria-label="Bildtyp">
        <div className="capture-chiprow">
          {SURFACES.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`capture-chip${
                surfaceKind === s.id ? ' capture-chip-active' : ''
              }`}
              onClick={() => setSurfaceKind(s.id)}
            >
              <span className="capture-chip-label">{s.label}</span>
              <span className="capture-chip-sub">{s.sub}</span>
            </button>
          ))}
        </div>
        <input
          type="text"
          className="capture-note"
          value={note}
          placeholder="Notiz für Mira (optional)"
          onChange={(e) => setNote(e.target.value)}
          maxLength={120}
        />
      </div>

      <section className="capture-stage">
        <div className="capture-viewer">
          <video
            ref={c.videoRef}
            className={`capture-video${
              c.status === 'streaming' || c.status === 'capturing'
                ? ' capture-video-live'
                : ''
            }`}
            playsInline
            muted
          />
          {c.status !== 'streaming' && c.status !== 'capturing' ? (
            <div className="capture-placeholder">
              <p>{stageMessage}</p>
            </div>
          ) : null}
          <canvas ref={c.canvasRef} className="capture-canvas" />
        </div>

        <div className="capture-buttons">
          {c.status === 'streaming' || c.status === 'capturing' ? (
            <>
              <button
                type="button"
                className="capture-shutter"
                onClick={() => void c.shutter()}
                disabled={c.status === 'capturing'}
              >
                Aufnahme
              </button>
              <button
                type="button"
                className="capture-secondary"
                onClick={() => c.stopCamera()}
              >
                Kamera stoppen
              </button>
            </>
          ) : (
            <>
              <button
                type="button"
                className="capture-shutter"
                onClick={() => void c.startCamera()}
                disabled={c.status === 'requesting' || c.status === 'processing'}
              >
                Kamera starten
              </button>
              <button
                type="button"
                className="capture-secondary"
                onClick={() => fileInputRef.current?.click()}
                disabled={c.status === 'processing'}
              >
                Bild hochladen
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => {
                  const f = e.target.files?.[0]
                  if (f) {
                    void c.uploadFile(f)
                    e.target.value = ''
                  }
                }}
              />
            </>
          )}
        </div>
        <p className="capture-status">{stageMessage}</p>
      </section>

      {c.lastResult ? (
        <section className="capture-result">
          <header className="capture-result-head">
            <p className="capture-result-eyebrow">
              {c.lastResult.engine === 'gemma_vision'
                ? `Mira (${c.lastResult.model || 'Gemma 4'})`
                : 'Mira (Skelett-Modus)'}
            </p>
            <p className="capture-result-time">
              {c.lastResult.duration_ms} ms
            </p>
          </header>
          <p className="capture-result-transcript">
            {c.lastResult.transcript || '(kein Text erkannt)'}
          </p>
          {c.lastResult.advice ? (
            <p className="capture-result-advice">{c.lastResult.advice}</p>
          ) : null}
          {c.lastResult.words.length > 0 ? (
            <ul className="capture-words">
              {c.lastResult.words.map((w, i) => (
                <li key={i} className="capture-word">
                  <p className="capture-word-surface">
                    {w.gender ? <em>{w.gender} </em> : null}
                    {w.surface}
                  </p>
                  {w.definition_de ? (
                    <p className="capture-word-de">{w.definition_de}</p>
                  ) : null}
                  {w.definition_en ? (
                    <p className="capture-word-en">{w.definition_en}</p>
                  ) : null}
                </li>
              ))}
            </ul>
          ) : null}
        </section>
      ) : null}

      {c.recent.length > 0 ? (
        <section className="capture-recent">
          <h2 className="capture-recent-title">Frühere Aufnahmen</h2>
          <ul>
            {c.recent.slice(0, 5).map((r) => (
              <li key={r.id}>
                <span className="capture-recent-kind">{r.surface_kind}</span>
                <span className="capture-recent-text">
                  {r.transcript_head || '(ohne Text)'}
                </span>
                <span className="capture-recent-time">
                  {prettyTime(r.captured_at)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  )
}

function prettyTime(iso: string): string {
  try {
    return new Date(iso).toLocaleString('de-DE', {
      day: 'numeric',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return ''
  }
}
