// Konversation room (PIVOT_ROADMAP §B.5).
//
// One big mic at the centre. Tap to record, tap to stop, listen.
// Transcript log builds beneath. The page picks up scenario +
// competency hints from the session store so deep links land here
// already primed.

'use client'

import { useEffect, useMemo, useRef, useState } from 'react'

import { useKonversation } from '@/hooks/useKonversation'

const SCENARIOS: { id: string; label: string }[] = [
  { id: 'Smalltalk', label: 'Smalltalk' },
  { id: 'Bäckerei', label: 'Bäckerei' },
  { id: 'Anamnesegespräch', label: 'Anamnesegespräch' },
  { id: 'Café', label: 'Café' },
]

function makeSessionId(): string {
  if (typeof window !== 'undefined' && window.crypto?.randomUUID) {
    return `kv-${window.crypto.randomUUID().slice(0, 12)}`
  }
  return `kv-${Math.random().toString(36).slice(2, 14)}`
}

export function Konversation(): JSX.Element {
  const sessionRef = useRef<string>(makeSessionId())
  const [scenario, setScenario] = useState<string>('Smalltalk')
  const [targetCefr, setTargetCefr] = useState<string>('B1')

  const k = useKonversation({
    sessionId: sessionRef.current,
    scenario,
    targetCefr,
  })

  useEffect(() => {
    void k.refresh()
    // refresh once at mount; no dependency on `k.refresh` to keep it stable
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const buttonLabel = useMemo(() => {
    switch (k.status) {
      case 'recording':
        return 'Aufnahme stoppen'
      case 'processing':
        return 'Mira hört dir zu …'
      case 'speaking':
        return 'Mira spricht'
      case 'error':
        return 'Erneut versuchen'
      default:
        return 'Tippen, dann sprechen'
    }
  }, [k.status])

  const buttonDisabled =
    k.status === 'processing' || k.status === 'speaking'

  return (
    <div className="konversation">
      <header className="konversation-header">
        <p className="konversation-eyebrow">Konversation</p>
        <h1 className="konversation-title">5 Minuten mit Mira</h1>
        <p className="konversation-sub">
          Sag, was du sagen willst. Mira spricht ruhig, korrigiert sanft, und
          wartet, bis du fertig bist.
        </p>
      </header>

      <div className="konversation-controls" role="group" aria-label="Szenario">
        <div className="konversation-chiprow">
          {SCENARIOS.map((s) => (
            <button
              key={s.id}
              type="button"
              className={`konversation-chip${
                scenario === s.id ? ' konversation-chip-active' : ''
              }`}
              onClick={() => setScenario(s.id)}
              disabled={k.status === 'recording'}
            >
              {s.label}
            </button>
          ))}
        </div>
        <div className="konversation-cefr">
          <span>Niveau</span>
          <select
            value={targetCefr}
            onChange={(e) => setTargetCefr(e.target.value)}
            disabled={k.status === 'recording'}
          >
            <option value="A2">A2</option>
            <option value="B1">B1</option>
            <option value="B2">B2</option>
            <option value="C1">C1</option>
          </select>
        </div>
      </div>

      <div className="konversation-stage">
        <button
          type="button"
          className={`konversation-mic konversation-mic-${k.status}`}
          onClick={() => void k.toggle()}
          disabled={buttonDisabled}
          aria-label={buttonLabel}
        >
          <span className="konversation-mic-glyph" aria-hidden="true" />
          <span className="konversation-mic-label">{buttonLabel}</span>
        </button>
        {k.status === 'recording' ? (
          <p className="konversation-hint">Tippe wieder, wenn du fertig bist.</p>
        ) : null}
        {k.error ? <p className="konversation-error">{k.error}</p> : null}
        {k.ttsUnavailable ? (
          <p className="konversation-warn">
            Piper offline — Browser-Stimme aktiv.
          </p>
        ) : null}
      </div>

      <section className="konversation-log" aria-live="polite">
        {k.log.length === 0 ? (
          <p className="konversation-empty">
            Noch nichts gesagt. Drück den Knopf und sprich frei.
          </p>
        ) : (
          k.log
            .slice(-10)
            .map((t) => (
              <article
                key={t.id}
                className={`konversation-bubble konversation-bubble-${t.role}`}
              >
                <header className="konversation-bubble-meta">
                  <span>
                    {t.role === 'assistant' ? 'Mira' : 'Du'}
                  </span>
                  <span className="konversation-bubble-time">
                    {prettyTime(t.occurred_at)}
                  </span>
                </header>
                <p className="konversation-bubble-text">
                  {t.text || (t.role === 'user' ? '(unverstanden)' : '')}
                </p>
              </article>
            ))
        )}
      </section>

      {k.lastTurn?.user_transcript?.text ? (
        <section className="konversation-score" aria-live="polite">
          <button
            type="button"
            className="konversation-score-cta"
            onClick={() => void k.score(k.lastTurn!.user_transcript.text)}
            disabled={k.scoring || k.status === 'recording'}
          >
            {k.scoring
              ? 'Aussprache wird geprüft …'
              : 'Aussprache bewerten'}
          </button>
          {k.lastScore ? <PronunciationCard score={k.lastScore} /> : null}
        </section>
      ) : null}

      {k.lastTurn ? (
        <footer className="konversation-meta">
          {k.lastTurn.duration_ms} ms · STT {k.lastTurn.stt_engine} ·{' '}
          {k.lastTurn.llm_model}
        </footer>
      ) : null}
    </div>
  )
}

function PronunciationCard({
  score,
}: {
  score: import('@/lib/types').PronunciationScore
}): JSX.Element {
  const overallPct =
    score.overall != null ? Math.round(score.overall * 100) : null
  return (
    <article className="konversation-score-card">
      <header className="konversation-score-head">
        <span className="konversation-score-chip">
          {overallPct != null ? `${overallPct}/100` : 'offline'}
        </span>
        <span className="konversation-score-engine">
          {score.engine === 'librosa' ? 'MFCC + DTW' : 'Skelett'}
        </span>
      </header>
      {score.segments.length > 0 ? (
        <ol className="konversation-score-strip">
          {score.segments.map((s, idx) => (
            <li
              key={`${s.word}-${idx}`}
              className={`konversation-score-seg konversation-score-seg-${tier(
                s.score,
              )}`}
              title={s.hint || `${Math.round(s.score * 100)}/100`}
            >
              <span className="konversation-score-word">{s.word}</span>
              <span className="konversation-score-bar">
                <span
                  className="konversation-score-fill"
                  style={{ width: `${Math.round(s.score * 100)}%` }}
                />
              </span>
            </li>
          ))}
        </ol>
      ) : null}
      {score.critique ? (
        <p className="konversation-score-critique">{score.critique}</p>
      ) : null}
    </article>
  )
}

function tier(score: number): 'green' | 'amber' | 'red' {
  if (score >= 0.75) return 'green'
  if (score >= 0.5) return 'amber'
  return 'red'
}

function prettyTime(iso: string): string {
  try {
    return new Date(iso).toLocaleTimeString('de-DE', {
      hour: '2-digit',
      minute: '2-digit',
    })
  } catch {
    return ''
  }
}
