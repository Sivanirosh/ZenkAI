'use client'

/**
 * DistrictCard — drawer that opens when a learner taps an Atlas district
 * (PIVOT_ROADMAP §B.4). Shows the capability sentence Mira derived from
 * the planner, a row of MasteryDials per competency, and three
 * "doors" (Lesen / Sprechen / Üben). Sprechen + Üben are intentionally
 * disabled in this slice — they light up after B.5 / B.6 ship the rooms.
 */

import { useEffect } from 'react'

import { MasteryDial } from '@/components/MasteryDial'
import type { AtlasDistrict } from '@/lib/types'

interface DistrictCardProps {
  district: AtlasDistrict
  onClose: () => void
  onAskMira: (door: 'reader' | 'voice' | 'capture' | 'drill') => void
  busy: boolean
}

const STATUS_LABEL: Record<AtlasDistrict['status'], string> = {
  mastered: 'gemeistert',
  current: 'diese Woche',
  queued: 'als Nächstes',
  locked: 'gesperrt',
}

export function DistrictCard({
  district,
  onClose,
  onAskMira,
  busy,
}: DistrictCardProps): JSX.Element {
  // Close on Escape — modal hygiene.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  return (
    <div
      className="district-card-backdrop"
      onClick={onClose}
      role="presentation"
    >
      <aside
        className="district-card"
        role="dialog"
        aria-label={district.label}
        onClick={(e) => e.stopPropagation()}
      >
        <header className="district-card-header">
          <div>
            <p className="district-card-eyebrow">
              Woche {district.week_index} · {STATUS_LABEL[district.status]}
            </p>
            <h2 className="district-card-title">{district.label}</h2>
          </div>
          <button
            type="button"
            className="district-card-close"
            onClick={onClose}
            aria-label="Schließen"
          >
            ×
          </button>
        </header>

        {district.capability_sentence ? (
          <p className="district-card-capability">
            „{district.capability_sentence}"
          </p>
        ) : null}

        <section className="district-card-mastery">
          {district.competencies.map((c) => (
            <MasteryDial
              key={c.competency_id}
              confidence={c.confidence}
              variance={c.variance}
              label={c.label}
              cefr={c.cefr}
              size={84}
            />
          ))}
        </section>

        <section className="district-card-doors">
          <button
            type="button"
            className="btn-primary district-card-door"
            onClick={() => onAskMira('reader')}
            disabled={busy || district.status === 'locked'}
          >
            Lesen
          </button>
          <button
            type="button"
            className="btn-secondary district-card-door"
            disabled
            title="Konversation kommt mit B.5"
          >
            Sprechen
          </button>
          <button
            type="button"
            className="btn-secondary district-card-door"
            disabled
            title="Drill kommt mit B.6 / B.10"
          >
            Üben
          </button>
        </section>
      </aside>
    </div>
  )
}
