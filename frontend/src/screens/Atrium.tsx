'use client'

import { useCallback, useEffect, useState } from 'react'

import { CapabilityChip } from '@/components/CapabilityChip'
import { MasteryDial } from '@/components/MasteryDial'
import { MiraStream } from '@/components/MiraStream'
import { useMira } from '@/hooks/useMira'
import { getAtrium } from '@/lib/api'
import type { AtriumPayload, ToolResult } from '@/lib/types'
import { useSession } from '@/store/session'

const FALLBACK: AtriumPayload = {
  greeting: 'willkommen',
  today_plan: [
    'einen Patientenfall auf B1 zusammenfassen',
    'die Wörter aus gestern wiederholen',
  ],
  doors: [
    { id: 'reader', label: 'Lesen', subtitle: 'zur nächsten Passage' },
    { id: 'voice', label: 'Sprechen', subtitle: '5 Minuten Mira' },
    { id: 'capture', label: 'Sehen', subtitle: 'die Welt fotografieren' },
    { id: 'atlas', label: 'Karte', subtitle: 'dein Lernplan' },
  ],
  mastery_top: [],
  generated_at: '',
}

/**
 * Atrium — the home screen the learner lands on (PIVOT_ROADMAP §8.2).
 *
 * Three layers, top to bottom:
 *   1. Header: serif greeting + the date.
 *   2. Capability strip: two CapabilityChips built from `today_plan`.
 *   3. Mastery row: three MasteryDials of the top competencies.
 *   4. Door grid: four squares (Lesen / Sprechen / Sehen / Karte). Lesen
 *      asks Mira for a paragraph recommendation, then jumps the Reader.
 *   5. MiraStream footer: live during streaming, otherwise quiet.
 */
export function Atrium(): JSX.Element {
  const [data, setData] = useState<AtriumPayload>(FALLBACK)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

  const setTab = useSession((s) => s.setTab)
  const setCurrentParagraph = useSession((s) => s.setCurrentParagraph)

  const mira = useMira()

  useEffect(() => {
    let cancelled = false
    const ac = new AbortController()
    setLoading(true)
    getAtrium({ signal: ac.signal })
      .then((payload) => {
        if (!cancelled) {
          setData(payload)
          setLoadError(null)
        }
      })
      .catch((err) => {
        if (cancelled) return
        if ((err as { name?: string }).name === 'AbortError') return
        setLoadError(String(err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
      ac.abort()
    }
  }, [])

  // When Mira returns a paragraph_id, jump to the Reader and seed it.
  // When she returns a room_id (start_conversation / start_capture),
  // jump to the matching room.
  useEffect(() => {
    if (mira.results.length === 0) return
    const newest = mira.results[mira.results.length - 1]
    if (isParagraph(newest)) {
      setCurrentParagraph(String(newest.paragraph_id))
      setTab('rd')
      return
    }
    const room = newest.room_id
    if (room === 'konversation') setTab('kv')
    else if (room === 'capture') setTab('cap')
  }, [mira.results, setCurrentParagraph, setTab])

  const askForReading = useCallback(() => {
    void mira.run({
      current_screen: 'reader',
      time_budget_min: 12,
      goal_text: 'die nächste sinnvolle Passage öffnen',
    })
  }, [mira])

  const askForCapability = useCallback(
    (sentence: string) => {
      void mira.run({
        current_screen: 'atrium',
        user_message: sentence,
        time_budget_min: 12,
      })
    },
    [mira],
  )

  const todayPlan = data.today_plan.length > 0 ? data.today_plan : FALLBACK.today_plan
  const doors = data.doors.length === 4 ? data.doors : FALLBACK.doors
  const top3 = data.mastery_top.slice(0, 3)

  return (
    <div className="atrium">
      <header className="atrium-header">
        <h1 className="atrium-greeting">
          {data.greeting}
          <span className="atrium-greeting-comma">,</span>
        </h1>
        <p className="atrium-date">{prettyDate()}</p>
        {loading ? <p className="atrium-loading">…</p> : null}
        {loadError ? (
          <p className="atrium-error">Atrium offline ({loadError}). Zeige Fallback.</p>
        ) : null}
      </header>

      <section className="atrium-capabilities">
        {todayPlan.slice(0, 2).map((sentence, i) => (
          <CapabilityChip
            key={i}
            sentence={sentence}
            onActivate={() => askForCapability(sentence)}
            disabled={mira.status === 'streaming'}
          />
        ))}
      </section>

      {top3.length > 0 ? (
        <section className="atrium-mastery">
          {top3.map((m) => (
            <MasteryDial
              key={m.competency_id}
              confidence={m.confidence}
              variance={m.variance}
              label={m.label}
              cefr={m.cefr ?? null}
            />
          ))}
        </section>
      ) : null}

      <section className="door-grid" aria-label="Türen">
        {doors.map((door) => {
          const handler =
            door.id === 'reader'
              ? askForReading
              : door.id === 'voice'
              ? () => setTab('kv')
              : door.id === 'capture'
              ? () => setTab('cap')
              : door.id === 'atlas'
              ? () => setTab('al')
              : undefined
          return (
            <button
              key={door.id}
              type="button"
              className={`door door-${door.id}${
                handler ? '' : ' door-disabled'
              }`}
              onClick={handler}
              disabled={!handler || mira.status === 'streaming'}
            >
              <span className="door-label">{door.label}</span>
              <span className="door-subtitle">{door.subtitle}</span>
            </button>
          )
        })}
      </section>

      <footer className="atrium-mira-footer">
        <MiraStream
          thoughts={mira.thoughts}
          intents={mira.intents}
          results={mira.results}
          errors={mira.errors}
          status={mira.status}
        />
      </footer>
    </div>
  )
}

function isParagraph(res: ToolResult): boolean {
  return typeof res.paragraph_id === 'string' && res.paragraph_id.length > 0
}

function prettyDate(): string {
  try {
    return new Intl.DateTimeFormat('de-DE', {
      weekday: 'long',
      day: 'numeric',
      month: 'long',
    }).format(new Date())
  } catch {
    return ''
  }
}
