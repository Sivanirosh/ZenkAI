// Mic button — idle / recording / processing states.

'use client'

interface Props {
  status?: 'idle' | 'recording' | 'processing' | 'error'
  unavailable?: boolean
  unavailableHint?: string | null
  disabled?: boolean
  onToggle?: () => void
}

export function MicButton({
  status = 'idle',
  unavailable = false,
  unavailableHint,
  disabled = false,
  onToggle,
}: Props) {
  const isDisabled = disabled || unavailable || status === 'processing'
  const recording = status === 'recording'
  const label = recording
    ? 'Aufnahme beenden'
    : status === 'processing'
      ? 'Verarbeiten\u2026'
      : 'Aufnahme starten'
  const title = unavailable
    ? (unavailableHint ??
        'STT (faster-whisper) ist nicht installiert — `pip install -e ".[voice]"`')
    : label
  const icon =
    status === 'processing'
      ? '\u25CB'
      : recording
        ? '\u23F9'
        : '\uD83C\uDFA4'

  return (
    <button
      className={`mic-btn${recording ? ' recording' : ''}`}
      type="button"
      disabled={isDisabled}
      onClick={onToggle}
      aria-label={label}
      title={title}
    >
      {icon}
    </button>
  )
}
