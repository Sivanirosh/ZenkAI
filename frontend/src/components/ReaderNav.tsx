// Compact, presentational navigation control for the Leseraum.
//
// Shown sticky-top (in "bar" mode) and sticky-bottom (in "footer" mode)
// so readers can jump chapters / pages without scrolling. All logic —
// resume, pagination, prefetch — lives in the Reader screen; this file
// only emits events.

'use client'

import { useCallback } from 'react'

export interface ReaderNavProps {
  chapter: number
  chapterCount: number
  pageIndex: number
  pageCount: number
  canPrev: boolean
  canNext: boolean
  onChapterChange: (chapter: number) => void
  onPageChange: (pageIndex: number) => void
  onPrev: () => void
  onNext: () => void
  variant: 'top' | 'bottom'
  fontScale?: number
  onFontScaleChange?: (scale: number) => void
  focusMode?: boolean
  onToggleFocus?: () => void
}

const FONT_SCALES = [0.9, 1, 1.15, 1.3]

export function ReaderNav(props: ReaderNavProps) {
  const {
    chapter,
    chapterCount,
    pageIndex,
    pageCount,
    canPrev,
    canNext,
    onChapterChange,
    onPageChange,
    onPrev,
    onNext,
    variant,
    fontScale,
    onFontScaleChange,
    focusMode,
    onToggleFocus,
  } = props

  const handleChapterSelect = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => {
      const value = Number.parseInt(e.target.value, 10)
      if (!Number.isNaN(value)) onChapterChange(value)
    },
    [onChapterChange],
  )

  const handleScrub = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number.parseInt(e.target.value, 10)
      if (!Number.isNaN(value)) onPageChange(value)
    },
    [onPageChange],
  )

  const handlePageInput = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const value = Number.parseInt(e.target.value, 10)
      if (Number.isNaN(value)) return
      const clamped = Math.min(Math.max(value - 1, 0), Math.max(pageCount - 1, 0))
      onPageChange(clamped)
    },
    [pageCount, onPageChange],
  )

  const chapters: number[] = []
  for (let i = 1; i <= chapterCount; i++) chapters.push(i)

  if (variant === 'top') {
    return (
      <div className="reader-nav reader-nav-top">
        <div className="reader-nav-group">
          <label className="reader-nav-label" htmlFor="rn-chapter">
            Kapitel
          </label>
          <select
            id="rn-chapter"
            className="reader-nav-select"
            value={chapter}
            onChange={handleChapterSelect}
          >
            {chapters.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <span className="reader-nav-muted">/ {chapterCount}</span>
        </div>

        <div className="reader-nav-group">
          <label className="reader-nav-label" htmlFor="rn-page">
            Seite
          </label>
          <input
            id="rn-page"
            className="reader-nav-page-input"
            type="number"
            min={1}
            max={Math.max(pageCount, 1)}
            value={pageCount === 0 ? 0 : pageIndex + 1}
            onChange={handlePageInput}
          />
          <span className="reader-nav-muted">/ {pageCount}</span>
        </div>

        <div className="reader-nav-spacer" />

        {onFontScaleChange ? (
          <div className="reader-nav-group reader-nav-fonts">
            {FONT_SCALES.map((s, i) => (
              <button
                key={s}
                type="button"
                className={
                  'reader-nav-font' +
                  (Math.abs((fontScale ?? 1) - s) < 0.001 ? ' active' : '')
                }
                onClick={() => onFontScaleChange(s)}
                aria-label={`Schriftgr\u00F6\u00DFe ${i + 1}`}
                title={`Schriftgr\u00F6\u00DFe ${i + 1}`}
              >
                A
                <span
                  style={{
                    fontSize: `${Math.round(s * 9)}px`,
                    marginLeft: 1,
                  }}
                >
                  A
                </span>
              </button>
            ))}
          </div>
        ) : null}

        {onToggleFocus ? (
          <button
            type="button"
            className={'reader-nav-focus' + (focusMode ? ' active' : '')}
            onClick={onToggleFocus}
            title="Fokus-Modus (F)"
            aria-pressed={focusMode ? 'true' : 'false'}
          >
            {focusMode ? 'Fokus an' : 'Fokus'}
          </button>
        ) : null}
      </div>
    )
  }

  // bottom variant: prev / scrubber / next
  return (
    <div className="reader-nav reader-nav-bottom">
      <button
        type="button"
        className="reader-nav-btn"
        onClick={onPrev}
        disabled={!canPrev}
        aria-label="Vorherige Seite"
      >
        {'\u2190'} Vorherige
      </button>

      <div className="reader-nav-scrub">
        <input
          type="range"
          min={0}
          max={Math.max(pageCount - 1, 0)}
          value={pageIndex}
          onChange={handleScrub}
          aria-label="Seiten-Scrubber"
        />
        <div className="reader-nav-scrub-label">
          Seite {pageCount === 0 ? 0 : pageIndex + 1} / {pageCount}
          {chapterCount > 1 ? (
            <span className="reader-nav-muted">
              {' '}
              {'\u00B7'} Kapitel {chapter} / {chapterCount}
            </span>
          ) : null}
        </div>
      </div>

      <button
        type="button"
        className="reader-nav-btn"
        onClick={onNext}
        disabled={!canNext}
        aria-label="N\u00E4chste Seite"
      >
        N{'\u00E4'}chste {'\u2192'}
      </button>
    </div>
  )
}
