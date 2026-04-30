'use client'

interface CapabilityChipProps {
  /** Single sentence: „einen Patientenfall auf B1 zusammenfassen". */
  sentence: string
  /** Optional eyebrow label rendered above the sentence. */
  eyebrow?: string
  /** Click handler — Atrium fires a useMira.run(...) call here. */
  onActivate?: () => void
  disabled?: boolean
}

/**
 * Pill-shaped capability statement. Single line of intent — the
 * antithesis of a gamified streak counter. Click to ask Mira to help
 * the learner do exactly this.
 */
export function CapabilityChip({
  sentence,
  eyebrow = 'Du kannst:',
  onActivate,
  disabled = false,
}: CapabilityChipProps): JSX.Element {
  return (
    <button
      type="button"
      className="capability-chip"
      onClick={onActivate}
      disabled={disabled}
      aria-label={`${eyebrow} ${sentence}`}
    >
      <span className="capability-chip-eyebrow">{eyebrow}</span>
      <span className="capability-chip-sentence">{sentence}</span>
    </button>
  )
}
