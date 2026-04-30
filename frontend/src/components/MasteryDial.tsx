'use client'

import type { CSSProperties } from 'react'

interface MasteryDialProps {
  /** 0..100 — Bayesian mu rendered as the outer arc. */
  confidence: number
  /** 0..100 — Bayesian sigma rendered as the soft halo band. */
  variance: number
  label: string
  cefr?: string | null
  size?: number
}

/**
 * Concentric SVG ring: outer arc shows confidence, a translucent band
 * around it shows uncertainty. No animations on the arc itself — the
 * value is the story; the calm visual matches Mira's mood.
 *
 * Color tier comes from the confidence:
 *   < 40 → amber (needs work)
 *   < 75 → mixed teal
 *   else → confident teal
 */
export function MasteryDial({
  confidence,
  variance,
  label,
  cefr,
  size = 96,
}: MasteryDialProps): JSX.Element {
  const pct = clamp(confidence, 0, 100)
  const varPct = clamp(variance, 0, 100)
  const radius = size / 2 - 8
  const circumference = 2 * Math.PI * radius
  const arc = (pct / 100) * circumference
  const haloWidth = 4 + (varPct / 100) * 6

  const tone =
    pct < 40 ? 'var(--amber-400)' :
    pct < 75 ? 'var(--teal-200)' :
    'var(--teal-400)'

  const haloColor =
    pct < 40 ? 'var(--amber-100)' : 'var(--teal-50)'

  const cx = size / 2
  const cy = size / 2

  const ringStyle: CSSProperties = {
    transition: 'stroke-dasharray 200ms ease-out',
  }

  return (
    <div className="mastery-dial" role="img" aria-label={`${label} ${pct}%`}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke={haloColor}
          strokeWidth={haloWidth}
          opacity={0.6}
        />
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke="var(--color-border)"
          strokeWidth={3}
        />
        <circle
          cx={cx}
          cy={cy}
          r={radius}
          fill="none"
          stroke={tone}
          strokeWidth={4}
          strokeLinecap="round"
          strokeDasharray={`${arc} ${circumference - arc}`}
          transform={`rotate(-90 ${cx} ${cy})`}
          style={ringStyle}
        />
        <text
          x="50%"
          y="50%"
          dominantBaseline="central"
          textAnchor="middle"
          className="mastery-dial-pct"
          style={{ fill: 'var(--color-text-primary)', fontSize: size * 0.22, fontWeight: 600 }}
        >
          {pct}
        </text>
      </svg>
      <div className="mastery-dial-meta">
        <span className="mastery-dial-label">{label}</span>
        {cefr ? <span className="mastery-dial-cefr">{cefr}</span> : null}
      </div>
    </div>
  )
}

function clamp(n: number, lo: number, hi: number): number {
  if (Number.isNaN(n)) return lo
  return Math.max(lo, Math.min(hi, n))
}
