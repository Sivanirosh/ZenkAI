// Library grid item — matches the .book-card mockup markup.

'use client'

import type { WorkWithProgress } from '@/lib/types'

interface Props {
  work: WorkWithProgress
  onOpen: (work: WorkWithProgress) => void
}

function progressCopy(work: WorkWithProgress): string {
  if (work.progress.total_words === 0) return 'Nicht begonnen'
  return `${work.progress.known_pct.toFixed(0)}% bekannt`
}

function fillColor(pct: number): string {
  if (pct >= 25) return '#1D9E75'
  return '#EF9F27'
}

export function BookCard({ work, onOpen }: Props) {
  const pct = work.progress.known_pct
  return (
    <button className="book-card" onClick={() => onOpen(work)} type="button">
      <div
        className="book-spine"
        style={{ background: work.spine_color ?? '#085041' }}
      >
        {work.epoch ? (
          <span
            className="book-epoch"
            style={{ color: work.epoch_color ?? '#9FE1CB' }}
          >
            {work.epoch}
          </span>
        ) : null}
      </div>
      <div className="book-meta">
        <div className="book-title">{work.title}</div>
        <div className="book-author">{work.author}</div>
        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{
              width: `${Math.min(pct, 100)}%`,
              background: fillColor(pct),
            }}
          />
        </div>
        <div className="book-info">{progressCopy(work)}</div>
      </div>
    </button>
  )
}
