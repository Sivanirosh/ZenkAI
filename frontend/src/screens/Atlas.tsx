'use client'

/**
 * Atlas — the literal map of the learner's curriculum (PIVOT_ROADMAP §B.4).
 *
 * Layout strategy: districts are arranged on concentric arcs by week_index.
 * Week 1 sits in the centre (the learner's "home district"), later weeks
 * on outer rings. Prerequisite edges render as solid lines between
 * districts; the current_route gets a thicker stroke and a CSS keyframe
 * pulse — the §8.1 „delightful animation".
 *
 * Tap a node → opens the DistrictCard drawer.
 */

import { useCallback, useEffect, useMemo, useState } from 'react'

import { DistrictCard } from '@/components/DistrictCard'
import { useMira } from '@/hooks/useMira'
import { MiraStream } from '@/components/MiraStream'
import { getAtlas } from '@/lib/api'
import type { AtlasDistrict, AtlasPayload, ToolResult } from '@/lib/types'
import { useSession } from '@/store/session'

interface DistrictNode {
  district: AtlasDistrict
  cx: number
  cy: number
  r: number
}

const VIEW_W = 720
const VIEW_H = 480
const CENTER_X = VIEW_W / 2
const CENTER_Y = VIEW_H / 2

export function Atlas(): JSX.Element {
  const [data, setData] = useState<AtlasPayload | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [openDistrictId, setOpenDistrictId] = useState<string | null>(null)

  const setTab = useSession((s) => s.setTab)
  const setCurrentParagraph = useSession((s) => s.setCurrentParagraph)
  const mira = useMira()

  useEffect(() => {
    let cancelled = false
    const ac = new AbortController()
    setLoading(true)
    setLoadError(null)
    getAtlas({ signal: ac.signal })
      .then((payload) => {
        if (cancelled) return
        setData(payload)
      })
      .catch((err) => {
        if (cancelled) return
        if ((err as { name?: string }).name === 'AbortError') return
        setLoadError(formatErr(err))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
      ac.abort()
    }
  }, [])

  // When Mira returns a paragraph_id (Lesen door), jump to the Reader.
  useEffect(() => {
    if (mira.results.length === 0) return
    const lastWithPid = [...mira.results]
      .reverse()
      .find((r) => typeof r.paragraph_id === 'string' && r.paragraph_id)
    if (!lastWithPid) return
    const pid = String((lastWithPid as ToolResult).paragraph_id)
    setCurrentParagraph(pid)
    setTab('rd')
  }, [mira.results, setCurrentParagraph, setTab])

  const nodes = useMemo<DistrictNode[]>(
    () => layoutDistricts(data?.districts ?? []),
    [data?.districts],
  )
  const nodeById = useMemo(() => {
    const map = new Map<string, DistrictNode>()
    for (const n of nodes) map.set(n.district.id, n)
    return map
  }, [nodes])

  const routeSet = useMemo(
    () => new Set(data?.current_route ?? []),
    [data?.current_route],
  )

  const openDistrict = useMemo(() => {
    if (!openDistrictId) return null
    return data?.districts.find((d) => d.id === openDistrictId) ?? null
  }, [openDistrictId, data?.districts])

  const askMiraForRoom = useCallback(
    (district: AtlasDistrict, door: 'reader' | 'voice' | 'capture' | 'drill') => {
      void mira.run({
        current_screen: 'atlas',
        time_budget_min: 12,
        extras: {
          district_id: district.id,
          competencies: district.competencies.map((c) => c.competency_id),
          door,
        },
        goal_text:
          door === 'reader'
            ? `eine Passage zum Bezirk „${district.label}" öffnen`
            : undefined,
      })
      setOpenDistrictId(null)
    },
    [mira],
  )

  if (loading && !data) {
    return (
      <div className="atlas">
        <p className="atlas-loading">Lade deinen Plan…</p>
      </div>
    )
  }

  if (loadError && !data) {
    return (
      <div className="atlas">
        <header className="atlas-header">
          <h1 className="atlas-title">Atlas</h1>
        </header>
        <div className="atlas-empty">
          <p>Es gibt noch keinen Plan ({loadError}).</p>
          <button
            type="button"
            className="btn-primary"
            onClick={() => setTab('on')}
          >
            Ziel formulieren
          </button>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="atlas">
        <header className="atlas-header">
          <h1 className="atlas-title">Atlas</h1>
        </header>
        <div className="atlas-empty">
          <p>Kein Plan vorhanden.</p>
          <button
            type="button"
            className="btn-primary"
            onClick={() => setTab('on')}
          >
            Ziel formulieren
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="atlas">
      <header className="atlas-header">
        <p className="atlas-eyebrow">Karte · {data.horizon}</p>
        <h1 className="atlas-title">Dein Lernplan</h1>
        {data.rationale ? (
          <p className="atlas-subhead">{data.rationale}</p>
        ) : null}
      </header>

      <div className="atlas-canvas-wrap">
        <svg
          className="atlas-svg"
          viewBox={`0 0 ${VIEW_W} ${VIEW_H}`}
          preserveAspectRatio="xMidYMid meet"
          role="img"
          aria-label="Atlas-Karte"
        >
          <defs>
            <radialGradient id="atlas-bg" cx="50%" cy="50%" r="65%">
              <stop offset="0%" stopColor="var(--teal-50)" stopOpacity="0.5" />
              <stop offset="100%" stopColor="var(--color-bg-secondary)" stopOpacity="0" />
            </radialGradient>
          </defs>
          <rect
            x={0}
            y={0}
            width={VIEW_W}
            height={VIEW_H}
            fill="url(#atlas-bg)"
          />

          {data.edges.map((edge, i) => {
            const [from, to] = edge
            const a = nodeById.get(from)
            const b = nodeById.get(to)
            if (!a || !b) return null
            const onRoute = routeSet.has(from) && routeSet.has(to)
            return (
              <line
                key={`edge-${i}`}
                x1={a.cx}
                y1={a.cy}
                x2={b.cx}
                y2={b.cy}
                className={`district-edge${onRoute ? ' district-edge-route' : ''}`}
              />
            )
          })}

          {nodes.map((node) => {
            const onRoute = routeSet.has(node.district.id)
            return (
              <DistrictNodeShape
                key={node.district.id}
                node={node}
                onRoute={onRoute}
                onClick={() => setOpenDistrictId(node.district.id)}
              />
            )
          })}
        </svg>

        <Legend />
      </div>

      {openDistrict ? (
        <DistrictCard
          district={openDistrict}
          onClose={() => setOpenDistrictId(null)}
          onAskMira={(door) => askMiraForRoom(openDistrict, door)}
          busy={mira.status === 'streaming'}
        />
      ) : null}

      <footer className="atlas-mira-footer">
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

interface DistrictNodeProps {
  node: DistrictNode
  onRoute: boolean
  onClick: () => void
}

function DistrictNodeShape({
  node,
  onRoute,
  onClick,
}: DistrictNodeProps): JSX.Element {
  const { district, cx, cy, r } = node
  const fillVar = statusFill(district.status, district.confidence)
  const strokeVar = statusStroke(district.status)
  const labelY = cy + r + 16
  return (
    <g
      className={`district-node district-node-${district.status}${
        onRoute ? ' district-node-route' : ''
      }`}
      onClick={onClick}
      tabIndex={0}
      role="button"
      aria-label={`${district.label}, ${district.status}`}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onClick()
        }
      }}
    >
      {onRoute ? (
        <circle
          cx={cx}
          cy={cy}
          r={r + 6}
          fill="none"
          stroke={strokeVar}
          strokeWidth={1.5}
          className="district-route-pulse"
        />
      ) : null}
      <circle
        cx={cx}
        cy={cy}
        r={r}
        fill={fillVar}
        stroke={strokeVar}
        strokeWidth={onRoute ? 3 : 2}
      />
      <text
        x={cx}
        y={cy}
        textAnchor="middle"
        dominantBaseline="central"
        className="district-node-confidence"
      >
        {district.confidence}
      </text>
      <text
        x={cx}
        y={labelY}
        textAnchor="middle"
        className="district-node-label"
      >
        {truncate(district.label, 18)}
      </text>
    </g>
  )
}

function Legend(): JSX.Element {
  return (
    <ul className="atlas-legend" aria-label="Legende">
      <li>
        <span className="atlas-legend-swatch atlas-legend-mastered" /> gemeistert
      </li>
      <li>
        <span className="atlas-legend-swatch atlas-legend-current" /> diese Woche
      </li>
      <li>
        <span className="atlas-legend-swatch atlas-legend-queued" /> als Nächstes
      </li>
      <li>
        <span className="atlas-legend-swatch atlas-legend-locked" /> gesperrt
      </li>
    </ul>
  )
}

// ─── Layout helpers ──────────────────────────────────────────

function layoutDistricts(districts: AtlasDistrict[]): DistrictNode[] {
  if (districts.length === 0) return []

  const byWeek = new Map<number, AtlasDistrict[]>()
  for (const d of districts) {
    const w = d.week_index || 1
    const bucket = byWeek.get(w) ?? []
    bucket.push(d)
    byWeek.set(w, bucket)
  }
  const weeks = [...byWeek.keys()].sort((a, b) => a - b)
  const baseRadius = 90
  const ringStep = 92

  const nodes: DistrictNode[] = []
  weeks.forEach((week, weekIdx) => {
    const bucket = byWeek.get(week) ?? []
    if (weekIdx === 0 && bucket.length === 1) {
      nodes.push({
        district: bucket[0],
        cx: CENTER_X,
        cy: CENTER_Y,
        r: 44,
      })
      return
    }
    const ringRadius = weekIdx === 0 ? baseRadius : baseRadius + weekIdx * ringStep
    const count = bucket.length
    bucket.forEach((district, i) => {
      const angle =
        weekIdx === 0
          ? (Math.PI * 2 * i) / count - Math.PI / 2
          : (Math.PI * 2 * i) / count - Math.PI / 2 + weekIdx * 0.18
      const cx = CENTER_X + ringRadius * Math.cos(angle)
      const cy = CENTER_Y + ringRadius * Math.sin(angle)
      const clampedCx = Math.max(60, Math.min(VIEW_W - 60, cx))
      const clampedCy = Math.max(50, Math.min(VIEW_H - 50, cy))
      nodes.push({
        district,
        cx: clampedCx,
        cy: clampedCy,
        r: 38,
      })
    })
  })
  return nodes
}

function statusFill(
  status: AtlasDistrict['status'],
  confidence: number,
): string {
  switch (status) {
    case 'mastered':
      return 'var(--teal-200)'
    case 'current':
      return confidence >= 50 ? 'var(--teal-50)' : 'var(--amber-50)'
    case 'queued':
      return 'var(--color-bg-secondary)'
    case 'locked':
    default:
      return 'var(--color-bg)'
  }
}

function statusStroke(status: AtlasDistrict['status']): string {
  switch (status) {
    case 'mastered':
      return 'var(--teal-400)'
    case 'current':
      return 'var(--amber-400)'
    case 'queued':
      return 'var(--color-border-strong)'
    case 'locked':
    default:
      return 'var(--color-border)'
  }
}

function truncate(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, n - 1)}…`
}

function formatErr(err: unknown): string {
  if (err instanceof Error) return err.message
  return String(err)
}
