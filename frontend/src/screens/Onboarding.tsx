'use client'

/**
 * Onboarding — the very first screen a learner sees (PIVOT_ROADMAP §B.1).
 *
 * Single textarea („Warum bist du hier?"), submit, then a confirmation
 * card showing what Mira understood + an „Atlas öffnen" CTA. The flow is
 * deliberately editorial (Lora serif headline, generous whitespace, no
 * progress bars) per §8.1.
 *
 * Backend contract:
 *   POST /onboarding/goal { raw_text, horizon } → { goal_id, parsed,
 *                                                   plan_id, plan }
 *
 * On success we persist {goalId, planId, hasGoal} to the session store so
 * the app shell stops forcing this screen on subsequent paints.
 */

import { useCallback, useState } from 'react'

import { submitOnboardingGoal } from '@/lib/api'
import type { OnboardingGoalResponse, ParsedGoal } from '@/lib/types'
import { useSession } from '@/store/session'

const PLACEHOLDER =
  'Zum Beispiel: „Ich bin Ärztin aus Brasilien und möchte in 90 Tagen die FSP bestehen — Anamnesegespräche und Aufklärungen sollen sicher klappen."'

const DOMAIN_LABEL: Record<string, string> = {
  medical: 'Medizin',
  academic: 'Akademisch',
  daily: 'Alltag',
  other: 'Allgemein',
}

export function Onboarding(): JSX.Element {
  const [text, setText] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<OnboardingGoalResponse | null>(null)

  const setOnboarding = useSession((s) => s.setOnboarding)
  const setTab = useSession((s) => s.setTab)

  const submit = useCallback(async () => {
    const raw = text.trim()
    if (raw.length === 0 || submitting) return
    setSubmitting(true)
    setError(null)
    try {
      const res = await submitOnboardingGoal(raw)
      setResult(res)
      setOnboarding({
        goalId: res.goal_id,
        planId: res.plan_id,
        hasGoal: true,
      })
    } catch (err) {
      setError(formatError(err))
    } finally {
      setSubmitting(false)
    }
  }, [text, submitting, setOnboarding])

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>): void => {
    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
      event.preventDefault()
      void submit()
    }
  }

  if (result) {
    return (
      <ConfirmationCard
        result={result}
        onContinue={() => setTab('al')}
        onRevise={() => {
          setResult(null)
          setText('')
        }}
      />
    )
  }

  return (
    <div className="onboarding">
      <header className="onboarding-header">
        <p className="onboarding-eyebrow">Mira</p>
        <h1 className="onboarding-headline">Warum bist du hier?</h1>
        <p className="onboarding-subhead">
          Erzähl mir kurz, was du auf Deutsch erreichen willst — und bis wann.
          Ich baue dir daraus eine Lernkarte.
        </p>
      </header>

      <section className="onboarding-form">
        <textarea
          className="onboarding-textarea"
          placeholder={PLACEHOLDER}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          rows={6}
          maxLength={4000}
          autoFocus
          disabled={submitting}
          aria-label="Dein Ziel"
        />
        <div className="onboarding-actions">
          <span className="onboarding-hint">
            ⌘/Ctrl + Enter zum Absenden
          </span>
          <button
            type="button"
            className="btn-primary onboarding-submit"
            onClick={() => void submit()}
            disabled={submitting || text.trim().length === 0}
          >
            {submitting ? 'Mira denkt nach…' : 'Plan erstellen'}
          </button>
        </div>
        {error ? <p className="onboarding-error">{error}</p> : null}
      </section>
    </div>
  )
}

interface ConfirmationProps {
  result: OnboardingGoalResponse
  onContinue: () => void
  onRevise: () => void
}

function ConfirmationCard({
  result,
  onContinue,
  onRevise,
}: ConfirmationProps): JSX.Element {
  const parsed: ParsedGoal = result.parsed
  const weekCount = result.plan?.weeks?.length ?? 0
  const districtCount = result.plan?.districts?.length ?? 0
  return (
    <div className="onboarding">
      <header className="onboarding-header">
        <p className="onboarding-eyebrow">Mira</p>
        <h1 className="onboarding-headline">So habe ich dich verstanden.</h1>
        <p className="onboarding-subhead">
          Dein {result.horizon}-Plan ist bereit — {weekCount} Wochen,{' '}
          {districtCount} Bezirke.
        </p>
      </header>

      <section className="onboarding-summary">
        <SummaryRow label="Bereich" value={DOMAIN_LABEL[parsed.domain] ?? parsed.domain} />
        <SummaryRow label="CEFR-Ziel" value={parsed.target_cefr} />
        <SummaryRow
          label="Frist"
          value={parsed.deadline_iso ?? 'kein festes Datum'}
        />
        {parsed.scenarios.length > 0 ? (
          <SummaryRow
            label="Szenarien"
            value={parsed.scenarios.slice(0, 4).join(' · ')}
          />
        ) : null}
        {parsed.motivations.length > 0 ? (
          <SummaryRow
            label="Motivation"
            value={parsed.motivations.slice(0, 3).join(' · ')}
          />
        ) : null}
      </section>

      <div className="onboarding-cta-row">
        <button type="button" className="btn-secondary" onClick={onRevise}>
          Nochmal formulieren
        </button>
        <button type="button" className="btn-primary" onClick={onContinue}>
          Atlas öffnen →
        </button>
      </div>
    </div>
  )
}

function SummaryRow({
  label,
  value,
}: {
  label: string
  value: string
}): JSX.Element {
  return (
    <div className="onboarding-summary-row">
      <span className="onboarding-summary-label">{label}</span>
      <span className="onboarding-summary-value">{value}</span>
    </div>
  )
}

function formatError(err: unknown): string {
  if (err instanceof Error) return err.message
  return String(err)
}
