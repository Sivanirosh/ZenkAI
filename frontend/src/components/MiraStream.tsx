'use client'

import type {
  ToolError,
  ToolIntent,
  ToolResult,
} from '../lib/types'
import type { MiraStatus } from '../hooks/useMira'

interface MiraStreamProps {
  thoughts: string[]
  intents: ToolIntent[]
  results: ToolResult[]
  errors?: ToolError[]
  status: MiraStatus
  className?: string
}

/**
 * Live event renderer for one Mira turn.
 *
 * Layout (top-down):
 *  - Thought paragraphs as soft serif text
 *  - One chip per tool intent
 *  - One result card per tool result (rendered with a tool-aware
 *    summariser so common shapes — recommend_text, update_mastery,
 *    start_drill — read like sentences instead of JSON dumps).
 *  - Inline errors at the bottom.
 *
 * Intentionally framework-free: no animations on first paint, just a
 * subtle opacity transition when new chunks land. Status drives the
 * "Mira tippt" indicator at the very bottom.
 */
export function MiraStream({
  thoughts,
  intents,
  results,
  errors = [],
  status,
  className = '',
}: MiraStreamProps): JSX.Element {
  const isStreaming = status === 'streaming'
  const isQuiet =
    !isStreaming &&
    thoughts.length === 0 &&
    intents.length === 0 &&
    results.length === 0

  return (
    <section className={`mira-stream ${className}`} aria-live="polite">
      {thoughts.length > 0 && (
        <div className="mira-thoughts">
          {thoughts.map((t, i) => (
            <p key={i} className="mira-thought">{t}</p>
          ))}
        </div>
      )}

      {intents.length > 0 && (
        <ul className="mira-intents">
          {intents.map((intent, i) => (
            <li key={i} className="mira-intent">
              <span className="mira-intent-name">{intent.name}</span>
              <span className="mira-intent-args">{summariseArgs(intent.args)}</span>
            </li>
          ))}
        </ul>
      )}

      {results.length > 0 && (
        <ul className="mira-results">
          {results.map((res, i) => (
            <li key={i} className="mira-result-card">
              <header className="mira-result-name">{res.name as string}</header>
              <p className="mira-result-summary">{summariseResult(res)}</p>
            </li>
          ))}
        </ul>
      )}

      {errors.length > 0 && (
        <ul className="mira-errors">
          {errors.map((err, i) => (
            <li key={i} className="mira-error">
              <strong>{err.tool}</strong> · {err.code}: {err.message}
            </li>
          ))}
        </ul>
      )}

      {isStreaming && (
        <div className="mira-typing" aria-label="Mira denkt nach">
          <span className="mira-typing-dot" />
          <span className="mira-typing-dot" />
          <span className="mira-typing-dot" />
        </div>
      )}

      {isQuiet && status === 'idle' && (
        <p className="mira-empty">Mira ist bereit.</p>
      )}
    </section>
  )
}

function summariseArgs(args: Record<string, unknown>): string {
  const keys = Object.keys(args).slice(0, 3)
  if (keys.length === 0) return ''
  return keys
    .map((k) => `${k}=${shortValue(args[k])}`)
    .join(' · ')
}

function summariseResult(res: ToolResult): string {
  switch (res.name) {
    case 'recommend_text': {
      const pid = (res.paragraph_id as string) ?? '?'
      const reason = (res.reason as string) ?? ''
      return `Empfohlen: ${pid}${reason ? ` — ${reason}` : ''}`
    }
    case 'update_mastery': {
      const cid = (res.competency_id as string) ?? '?'
      const conf = res.confidence ?? '?'
      return `${cid} → confidence ${conf}/100`
    }
    case 'start_drill': {
      const cid = (res.competency_id as string) ?? '?'
      const count = (res.count as number) ?? 0
      return `Drill ${cid} (${count} Übungen)`
    }
    default: {
      const { name: _name, ...rest } = res
      return shortValue(rest)
    }
  }
}

function shortValue(v: unknown): string {
  if (v === null || v === undefined) return '∅'
  if (typeof v === 'string') return v.length > 40 ? `${v.slice(0, 40)}…` : v
  if (typeof v === 'number' || typeof v === 'boolean') return String(v)
  try {
    const json = JSON.stringify(v)
    return json.length > 60 ? `${json.slice(0, 60)}…` : json
  } catch {
    return '[obj]'
  }
}
