// Bibliothek screen: metrics + filter chips + book grid.
//
// Renders from the cached library payload instantly on mount; revalidates
// in the background. Tab-switching back into the library is therefore ~0 ms.

'use client'

import { useCallback, useMemo, useState } from 'react'
import * as api from '@/lib/api'
import type { WorkWithProgress } from '@/lib/types'
import { BookCard } from '@/components/BookCard'
import { useSession } from '@/store/session'
import { useCachedResource } from '@/hooks/useCachedResource'

const EPOCHS = ['Romantik', 'Expressionismus', 'Moderne', 'Volksliteratur'] as const

export function Library() {
  const openWork = useSession((s) => s.openWork)
  const epochFilter = useSession((s) => s.epochFilter)
  const setEpochFilter = useSession((s) => s.setEpochFilter)
  const libraryCache = useSession((s) => s.libraryCache)
  const setLibraryCache = useSession((s) => s.setLibraryCache)
  const setChaptersCache = useSession((s) => s.setChaptersCache)

  const [search, setSearch] = useState('')
  const [openError, setOpenError] = useState<string | null>(null)

  const fetchLibrary = useCallback(async (signal: AbortSignal) => {
    const [works, counts] = await Promise.all([
      api.listWorks({ signal }),
      api.vocabCounts({ signal }),
    ])
    return { works, counts, fetchedAt: Date.now() }
  }, [])

  const { data, loading, error, refresh } = useCachedResource({
    key: 'library',
    cached: libraryCache,
    fetcher: fetchLibrary,
    onData: setLibraryCache,
  })

  const works = data?.works ?? null
  const counts = data?.counts ?? null

  const filtered = useMemo(() => {
    if (!works) return []
    return works.filter((w) => {
      if (epochFilter && w.epoch !== epochFilter) return false
      if (search.trim()) {
        const needle = search.toLowerCase()
        if (
          !w.title.toLowerCase().includes(needle) &&
          !w.author.toLowerCase().includes(needle)
        )
          return false
      }
      return true
    })
  }, [works, epochFilter, search])

  const handleOpen = async (work: WorkWithProgress) => {
    try {
      const chapters = await api.listChapters(work.id)
      setChaptersCache(work.id, chapters)
      const first = chapters[0]?.first_paragraph_id ?? null
      openWork(work.id, first)
    } catch (err) {
      setOpenError(err instanceof Error ? err.message : String(err))
    }
  }

  const metrics = counts ?? {
    queued: 0,
    known: 0,
    exported: 0,
  }
  const activeBooks = works?.filter((w) => w.progress.total_words > 0).length ?? 0
  const fetchError = error ?? openError

  return (
    <div>
      <div className="search-row">
        <input
          type="text"
          placeholder="Suchen..."
          style={{ flex: 1, minWidth: 130 }}
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <div className="filter-chips">
          <button
            className={`chip${epochFilter === null ? ' selected' : ''}`}
            type="button"
            onClick={() => setEpochFilter(null)}
          >
            Alle
          </button>
          {EPOCHS.map((e) => (
            <button
              key={e}
              className={`chip${epochFilter === e ? ' selected' : ''}`}
              type="button"
              onClick={() => setEpochFilter(epochFilter === e ? null : e)}
            >
              {e}
            </button>
          ))}
        </div>
      </div>

      <div className="metrics">
        <div className="metric-card">
          <div className="metric-label">Bekannte W{'\u00F6'}rter</div>
          <div className="metric-value">{metrics.known.toLocaleString('de-DE')}</div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Gelernte W{'\u00F6'}rter</div>
          <div className="metric-value">
            {(metrics.queued + metrics.exported).toLocaleString('de-DE')}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-label">Aktive B{'\u00FC'}cher</div>
          <div className="metric-value">{activeBooks}</div>
        </div>
      </div>

      {fetchError ? (
        <div className="empty-state">
          <div style={{ marginBottom: 10 }}>Backend nicht erreichbar.</div>
          <div
            style={{
              fontSize: 12,
              color: 'var(--color-text-secondary)',
              marginBottom: 12,
            }}
          >
            {fetchError.startsWith('500') ||
            fetchError.toLowerCase().includes('failed to fetch')
              ? 'Die Verbindung zum FastAPI-Server wurde unterbrochen (oder der Server l\u00E4uft nicht).'
              : fetchError}
          </div>
          <div style={{ marginBottom: 12 }}>
            Starte das Backend mit{' '}
            <code>./scripts/run_backend.sh</code>
          </div>
          <button
            className="btn-primary"
            type="button"
            onClick={() => {
              setOpenError(null)
              refresh()
            }}
          >
            Erneut versuchen
          </button>
        </div>
      ) : null}

      {works === null && loading ? (
        <div className="skeleton">L{'\u00E4'}dt Bibliothek{'\u2026'}</div>
      ) : works && filtered.length === 0 ? (
        <div className="empty-state">
          Keine B{'\u00FC'}cher. F{'\u00FC'}hre{' '}
          <code>python scripts/ingest_book.py --gutenberg-id 22367</code> aus.
        </div>
      ) : works ? (
        <div className="book-grid">
          {filtered.map((work) => (
            <BookCard key={work.id} work={work} onOpen={handleOpen} />
          ))}
        </div>
      ) : null}
    </div>
  )
}
