// Leseraum: paginated, long-reading-friendly view over a work.
//
// Paragraphs are grouped client-side into "pages" of ~350 words (3-10
// paragraphs each). The reader navigates by page, not paragraph, with
// a sticky top bar (chapter + page + font + focus toggle) and a sticky
// bottom bar (prev / scrubber / next). Keyboard shortcuts, scroll
// restoration, fade transitions, and aggressive prefetching make page
// turns feel instant.

'use client'

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react'
import * as api from '@/lib/api'
import type {
  Chapter,
  Paragraph,
  ParagraphWithTokens,
  VocabCounts,
} from '@/lib/types'
import { TtsBar } from '@/components/TtsBar'
import { WordSpan } from '@/components/WordSpan'
import { ReaderNav } from '@/components/ReaderNav'
import {
  chapterKey,
  pageScrollKey,
  useSession,
} from '@/store/session'
import { useVocab } from '@/hooks/useVocab'
import { useCachedResource } from '@/hooks/useCachedResource'
import { buildPages, findPageIndex, type Page } from '@/lib/pagination'

interface ParagraphViewProps {
  paragraph: ParagraphWithTokens
  active?: boolean
}

// Render a paragraph by walking its word tokens in order and slicing the
// original text at their char offsets. Everything between two adjacent
// word ranges (punctuation, whitespace, dashes, quotes) is emitted as
// plain text — paragraphs.text is the single source of truth for typography.
// Runs in O(N) per paragraph with no intermediate strings.
function ParagraphView({ paragraph, active = false }: ParagraphViewProps) {
  const className = active ? 'tts-active' : undefined
  const { text, tokens, id } = paragraph

  if (tokens.length === 0) {
    return <p className={className}>{text}</p>
  }

  const nodes: React.ReactNode[] = []
  let cursor = 0
  for (let i = 0; i < tokens.length; i++) {
    const t = tokens[i]
    if (t.char_start > cursor) {
      nodes.push(text.slice(cursor, t.char_start))
    }
    nodes.push(
      <WordSpan
        key={`${t.char_start}-${i}`}
        token={t}
        sentence={text}
        paragraphId={id}
      />,
    )
    cursor = t.char_end
  }
  if (cursor < text.length) nodes.push(text.slice(cursor))

  return <p className={className}>{nodes}</p>
}

export function Reader() {
  const currentWorkId = useSession((s) => s.currentWorkId)
  const currentParagraphId = useSession((s) => s.currentParagraphId)
  const setCurrentParagraph = useSession((s) => s.setCurrentParagraph)
  const setTab = useSession((s) => s.setTab)

  const libraryCache = useSession((s) => s.libraryCache)
  const chaptersCache = useSession((s) => s.chaptersCache)
  const setChaptersCache = useSession((s) => s.setChaptersCache)
  const chapterParagraphsCache = useSession((s) => s.chapterParagraphsCache)
  const setChapterParagraphsCache = useSession(
    (s) => s.setChapterParagraphsCache,
  )
  const paragraphCache = useSession((s) => s.paragraphCache)
  const cacheParagraph = useSession((s) => s.cacheParagraph)

  const readerPrefs = useSession((s) => s.readerPrefs)
  const setReaderPrefs = useSession((s) => s.setReaderPrefs)
  const lastScrollByPage = useSession((s) => s.lastScrollByPage)
  const setPageScroll = useSession((s) => s.setPageScroll)

  const { bulkSet, statusMap: vocabStatusMap } = useVocab()

  // ── Vocab queue counts for the header pill ───────────
  const [vocabCounts, setVocabCounts] = useState<VocabCounts>({
    queued: 0,
    known: 0,
    exported: 0,
  })
  useEffect(() => {
    let cancelled = false
    api
      .vocabCounts()
      .then((c) => {
        if (!cancelled) setVocabCounts(c)
      })
      .catch(() => {
        /* best-effort */
      })
    return () => {
      cancelled = true
    }
    // Refresh on status-map mutation (queue or mark-known).
  }, [vocabStatusMap])

  const work = useMemo(
    () => libraryCache?.works.find((w) => w.id === currentWorkId) ?? null,
    [libraryCache, currentWorkId],
  )

  // ── Chapters ─────────────────────────────────────────
  const chaptersKey = currentWorkId ? `chapters:${currentWorkId}` : null
  useCachedResource<Chapter[]>({
    key: chaptersKey,
    cached: currentWorkId ? chaptersCache[currentWorkId] : undefined,
    fetcher: useCallback(
      (signal: AbortSignal) => api.listChapters(currentWorkId!, { signal }),
      [currentWorkId],
    ),
    onData: useCallback(
      (chapters: Chapter[]) => {
        if (currentWorkId) setChaptersCache(currentWorkId, chapters)
      },
      [currentWorkId, setChaptersCache],
    ),
  })
  const chapters = currentWorkId ? chaptersCache[currentWorkId] ?? null : null

  // ── Determine current chapter ────────────────────────
  // We derive the current chapter from either the current paragraph's
  // chapter (once loaded) or the stored paragraph's chapter guess. We
  // keep a sticky fallback while a page transition is in flight so the
  // nav doesn't flicker.
  const loadedCurrentParagraph = currentParagraphId
    ? paragraphCache[currentParagraphId] ?? null
    : null

  const stickyChapterRef = useRef<number | null>(null)
  const currentChapter =
    loadedCurrentParagraph?.chapter ??
    stickyChapterRef.current ??
    chapters?.[0]?.chapter ??
    1
  if (loadedCurrentParagraph) {
    stickyChapterRef.current = loadedCurrentParagraph.chapter
  }

  // ── Paragraph list for current chapter ───────────────
  const chapterParasKey =
    currentWorkId != null ? chapterKey(currentWorkId, currentChapter) : null

  useCachedResource<Paragraph[]>({
    key: chapterParasKey,
    cached: chapterParasKey
      ? chapterParagraphsCache[chapterParasKey]
      : undefined,
    fetcher: useCallback(
      (signal: AbortSignal) =>
        api.listChapterParagraphs(currentWorkId!, currentChapter, { signal }),
      [currentWorkId, currentChapter],
    ),
    onData: useCallback(
      (list: Paragraph[]) => {
        if (currentWorkId)
          setChapterParagraphsCache(currentWorkId, currentChapter, list)
      },
      [currentWorkId, currentChapter, setChapterParagraphsCache],
    ),
  })
  const chapterParagraphs: Paragraph[] = chapterParasKey
    ? chapterParagraphsCache[chapterParasKey] ?? []
    : []

  // ── Build pages for current chapter ─────────────────
  const pages: Page[] = useMemo(
    () => buildPages(chapterParagraphs),
    [chapterParagraphs],
  )

  const currentPageIdx = useMemo(() => {
    if (!currentParagraphId || pages.length === 0) return 0
    const idx = findPageIndex(pages, currentParagraphId)
    return idx < 0 ? 0 : idx
  }, [pages, currentParagraphId])

  const currentPage: Page | null =
    pages.length > 0 ? pages[Math.min(currentPageIdx, pages.length - 1)] : null

  // ── Fetch every paragraph of the current page ───────
  const [paragraphError, setParagraphError] = useState<string | null>(null)
  const [paragraphsLoading, setParagraphsLoading] = useState(false)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    if (!currentPage) return
    const missing = currentPage.paragraphIds.filter(
      (id) => !paragraphCache[id],
    )
    if (missing.length === 0) {
      setParagraphError(null)
      setParagraphsLoading(false)
      return
    }
    const ac = new AbortController()
    setParagraphError(null)
    setParagraphsLoading(true)
    api
      .getParagraphs(missing, { signal: ac.signal })
      .then(async (results) => {
        if (ac.signal.aborted) return
        const wordIds: string[] = []
        for (const p of results) {
          cacheParagraph(p)
          for (const t of p.tokens) {
            if (t.word_id) wordIds.push(t.word_id)
          }
        }
        if (wordIds.length > 0) {
          try {
            const { statuses } = await api.vocabStatusMap(
              Array.from(new Set(wordIds)),
              { signal: ac.signal },
            )
            if (!ac.signal.aborted) bulkSet(statuses)
          } catch {
            /* best-effort hydration */
          }
        }
        if (!ac.signal.aborted) setParagraphsLoading(false)
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return
        if (err instanceof Error && err.name === 'AbortError') return
        setParagraphError(err instanceof Error ? err.message : String(err))
        setParagraphsLoading(false)
      })
    return () => ac.abort()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    currentPage?.index,
    currentPage?.paragraphIds.join('|'),
    nonce,
    cacheParagraph,
    bulkSet,
  ])

  const pageParagraphs: ParagraphWithTokens[] = currentPage
    ? currentPage.paragraphIds
        .map((id) => paragraphCache[id])
        .filter((p): p is ParagraphWithTokens => Boolean(p))
    : []
  const pageReady =
    currentPage != null &&
    pageParagraphs.length === currentPage.paragraphIds.length

  // ── Prefetch next page's paragraphs ──────────────────
  useEffect(() => {
    if (!currentPage || !pages.length) return
    const next = pages[currentPage.index + 1]
    if (!next) return
    const missing = next.paragraphIds.filter((id) => !paragraphCache[id])
    if (missing.length === 0) return
    const ac = new AbortController()
    api
      .getParagraphs(missing, { signal: ac.signal })
      .then((results) => {
        for (const p of results) cacheParagraph(p)
      })
      .catch(() => {
        /* best-effort */
      })
    return () => ac.abort()
  }, [currentPage, pages, paragraphCache, cacheParagraph])

  // ── Prefetch neighbour chapters' paragraph lists ────
  useEffect(() => {
    if (!currentWorkId || !chapters) return
    const ac = new AbortController()
    const neighbours = [currentChapter - 1, currentChapter + 1]
    for (const c of neighbours) {
      if (c < 1) continue
      if (!chapters.some((ch) => ch.chapter === c)) continue
      const key = chapterKey(currentWorkId, c)
      if (chapterParagraphsCache[key]) continue
      api
        .listChapterParagraphs(currentWorkId, c, { signal: ac.signal })
        .then((list) => setChapterParagraphsCache(currentWorkId, c, list))
        .catch(() => {
          /* best-effort */
        })
    }
    return () => ac.abort()
  }, [
    currentWorkId,
    currentChapter,
    chapters,
    chapterParagraphsCache,
    setChapterParagraphsCache,
  ])

  // ── Scroll restore / save ────────────────────────────
  const textRef = useRef<HTMLDivElement | null>(null)
  const scrollKey =
    currentWorkId && currentPage
      ? pageScrollKey(currentWorkId, currentChapter, currentPage.index)
      : null

  useLayoutEffect(() => {
    if (!scrollKey || !pageReady) return
    const saved = lastScrollByPage[scrollKey] ?? 0
    window.scrollTo({ top: saved, behavior: 'auto' })
  }, [scrollKey, pageReady, lastScrollByPage])

  useEffect(() => {
    if (!scrollKey) return
    let ticking = false
    const handler = () => {
      if (ticking) return
      ticking = true
      window.requestAnimationFrame(() => {
        setPageScroll(scrollKey, window.scrollY)
        ticking = false
      })
    }
    window.addEventListener('scroll', handler, { passive: true })
    return () => window.removeEventListener('scroll', handler)
  }, [scrollKey, setPageScroll])

  // ── Navigation helpers ──────────────────────────────
  const goToPage = useCallback(
    (idx: number) => {
      if (pages.length === 0) return
      const clamped = Math.min(Math.max(idx, 0), pages.length - 1)
      const target = pages[clamped]
      if (!target) return
      setCurrentParagraph(target.paragraphIds[0])
    },
    [pages, setCurrentParagraph],
  )

  const goToChapter = useCallback(
    (chapter: number) => {
      if (!chapters) return
      const ch = chapters.find((c) => c.chapter === chapter)
      if (!ch) return
      setCurrentParagraph(ch.first_paragraph_id)
    },
    [chapters, setCurrentParagraph],
  )

  const goPrevPage = useCallback(() => {
    if (!currentPage) return
    if (currentPage.index > 0) {
      goToPage(currentPage.index - 1)
      return
    }
    if (!chapters || !currentWorkId) return
    const prevChapter = currentChapter - 1
    const prevCh = chapters.find((c) => c.chapter === prevChapter)
    if (!prevCh) return
    const key = chapterKey(currentWorkId, prevChapter)
    const cachedList = chapterParagraphsCache[key]
    if (cachedList && cachedList.length > 0) {
      const prevPages = buildPages(cachedList)
      const last = prevPages[prevPages.length - 1]
      if (last) {
        setCurrentParagraph(last.paragraphIds[0])
        return
      }
    }
    setCurrentParagraph(prevCh.first_paragraph_id)
  }, [
    currentPage,
    goToPage,
    chapters,
    currentWorkId,
    currentChapter,
    chapterParagraphsCache,
    setCurrentParagraph,
  ])

  const goNextPage = useCallback(() => {
    if (!currentPage) return
    if (currentPage.index < pages.length - 1) {
      goToPage(currentPage.index + 1)
      return
    }
    if (!chapters) return
    const nextCh = chapters.find((c) => c.chapter === currentChapter + 1)
    if (!nextCh) return
    setCurrentParagraph(nextCh.first_paragraph_id)
  }, [currentPage, pages, goToPage, chapters, currentChapter, setCurrentParagraph])

  const canPrev =
    (currentPage?.index ?? 0) > 0 ||
    (chapters?.some((c) => c.chapter === currentChapter - 1) ?? false)
  const canNext =
    (currentPage && currentPage.index < pages.length - 1) ||
    (chapters?.some((c) => c.chapter === currentChapter + 1) ?? false)

  // ── Keyboard shortcuts ──────────────────────────────
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return
      }
      if (e.key === 'ArrowLeft') {
        if (e.shiftKey) {
          goToChapter(currentChapter - 1)
        } else {
          goPrevPage()
        }
        e.preventDefault()
      } else if (e.key === 'ArrowRight') {
        if (e.shiftKey) {
          goToChapter(currentChapter + 1)
        } else {
          goNextPage()
        }
        e.preventDefault()
      } else if (e.key === 'f' || e.key === 'F') {
        setReaderPrefs({ focusMode: !readerPrefs.focusMode })
        e.preventDefault()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [
    currentChapter,
    goPrevPage,
    goNextPage,
    goToChapter,
    setReaderPrefs,
    readerPrefs.focusMode,
  ])

  // ── Focus-mode body class ───────────────────────────
  useEffect(() => {
    if (typeof document === 'undefined') return
    const cls = 'reader-focus-mode'
    if (readerPrefs.focusMode) document.body.classList.add(cls)
    else document.body.classList.remove(cls)
    return () => document.body.classList.remove(cls)
  }, [readerPrefs.focusMode])

  // ── TTS chunks for the current page ──────────────────
  // paragraphs.text is already the exact text to speak; no reconstruction
  // from tokens is needed (and would have dropped punctuation).
  const ttsChunks = useMemo(
    () => pageParagraphs.map((p) => ({ id: p.id, text: p.text })),
    [pageParagraphs],
  )

  const [activeTtsId, setActiveTtsId] = useState<string | null>(null)
  useEffect(() => {
    setActiveTtsId(null)
  }, [currentPage?.index])

  // ── Reading progress (across the whole book) ────────
  const chapterCount = chapters?.length ?? 1
  const bookProgress = useMemo(() => {
    if (!chapters || chapters.length === 0 || pages.length === 0) return 0
    const chapterIndex = Math.max(
      chapters.findIndex((c) => c.chapter === currentChapter),
      0,
    )
    const within = pages.length > 0 ? currentPageIdx / pages.length : 0
    return Math.min((chapterIndex + within) / chapters.length, 1)
  }, [chapters, currentChapter, currentPageIdx, pages.length])

  // ── Early returns ───────────────────────────────────
  if (!currentWorkId) {
    return (
      <div className="empty-state">
        W{'\u00E4'}hle ein Buch in der Bibliothek.
        <div style={{ marginTop: 12 }}>
          <button
            className="btn-primary"
            type="button"
            onClick={() => setTab('lib')}
          >
            Zur Bibliothek
          </button>
        </div>
      </div>
    )
  }

  if (paragraphError && !pageReady) {
    return (
      <div className="empty-state">
        <div style={{ marginBottom: 10 }}>
          Fehler beim Laden: {paragraphError}
        </div>
        <button
          className="btn-primary"
          type="button"
          onClick={() => setNonce((n) => n + 1)}
        >
          Erneut versuchen
        </button>
      </div>
    )
  }

  return (
    <div
      className={'reader' + (readerPrefs.focusMode ? ' focus' : '')}
      style={
        {
          ['--reader-font-scale' as string]: readerPrefs.fontScale,
        } as React.CSSProperties
      }
    >
      <div
        className="reader-progress"
        role="progressbar"
        aria-valuenow={Math.round(bookProgress * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className="reader-progress-fill"
          style={{ width: `${bookProgress * 100}%` }}
        />
      </div>

      {!readerPrefs.focusMode ? (
        <div className="reader-header">
          <div>
            <div className="reader-work">
              {work?.author ?? ''}
              {work?.year ? ` \u00b7 ${work.year}` : ''}
            </div>
            <div className="reader-title">{work?.title ?? ''}</div>
          </div>
          <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
            <span
              className="badge"
              style={{
                background: 'var(--amber-50)',
                color: 'var(--amber-800)',
              }}
              title="In der Export-Warteschlange"
            >
              {vocabCounts.queued}{' '}
              {vocabCounts.queued === 1 ? 'Wort' : 'W\u00F6rter'} in
              Warteschlange
            </span>
            <button
              className="btn-secondary"
              type="button"
              onClick={() => setTab('ex')}
              disabled={vocabCounts.queued === 0}
              title={
                vocabCounts.queued === 0
                  ? 'Nichts zu exportieren'
                  : 'CSV exportieren'
              }
            >
              Export CSV
            </button>
            {paragraphsLoading ? (
              <span
                className="badge"
                style={{ color: 'var(--color-text-secondary)' }}
              >
                {'\u2026'}
              </span>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="reader-nav-sticky-top">
        <ReaderNav
          variant="top"
          chapter={currentChapter}
          chapterCount={chapterCount}
          pageIndex={currentPageIdx}
          pageCount={pages.length}
          canPrev={canPrev}
          canNext={canNext}
          onChapterChange={goToChapter}
          onPageChange={goToPage}
          onPrev={goPrevPage}
          onNext={goNextPage}
          fontScale={readerPrefs.fontScale}
          onFontScaleChange={(s) => setReaderPrefs({ fontScale: s })}
          focusMode={readerPrefs.focusMode}
          onToggleFocus={() =>
            setReaderPrefs({ focusMode: !readerPrefs.focusMode })
          }
        />
      </div>

      {!readerPrefs.focusMode ? (
        <div className="legend">
          <div className="legend-item">
            <div
              className="swatch"
              style={{
                background: 'var(--amber-50)',
                border: '0.5px solid var(--amber-100)',
              }}
            />
            Neu {'\u2014'} klicken = markieren
          </div>
          <div className="legend-item">
            <div
              style={{
                width: 14,
                height: 0,
                borderBottom: '2.5px solid var(--amber-600, #b45309)',
                display: 'inline-block',
              }}
            />
            Markiert {'\u2014'} im Export
          </div>
          <div className="legend-item">
            <div
              className="swatch"
              style={{ background: 'var(--color-bg-secondary)' }}
            />
            Bekannt {'\u2014'} Shift-Klick
          </div>
        </div>
      ) : null}

      <div
        ref={textRef}
        key={currentPage?.index ?? 'empty'}
        className="literary-text reader-page"
      >
        {pageReady ? (
          pageParagraphs.map((p) => (
            <ParagraphView
              key={p.id}
              paragraph={p}
              active={p.id === activeTtsId}
            />
          ))
        ) : (
          <div className="skeleton">
            L{'\u00E4'}dt Seite{'\u2026'}
          </div>
        )}
      </div>

      {!readerPrefs.focusMode ? (
        <div
          style={{
            display: 'flex',
            gap: 6,
            marginBottom: 12,
            flexWrap: 'wrap',
          }}
        >
          <span
            className="badge"
            style={{
              background: 'var(--amber-50)',
              color: 'var(--amber-800)',
            }}
            title="In der Export-Warteschlange"
          >
            {vocabCounts.queued} markiert
          </span>
          <span className="badge badge-teal">
            {vocabCounts.known} bekannt
          </span>
          {vocabCounts.exported > 0 ? (
            <span
              className="badge"
              style={{
                background: 'var(--color-bg-secondary)',
                color: 'var(--color-text-secondary)',
              }}
            >
              {vocabCounts.exported} exportiert
            </span>
          ) : null}
        </div>
      ) : null}

      <div className="reader-nav-sticky-bottom">
        <ReaderNav
          variant="bottom"
          chapter={currentChapter}
          chapterCount={chapterCount}
          pageIndex={currentPageIdx}
          pageCount={pages.length}
          canPrev={canPrev}
          canNext={canNext}
          onChapterChange={goToChapter}
          onPageChange={goToPage}
          onPrev={goPrevPage}
          onNext={goNextPage}
        />
      </div>

      {!readerPrefs.focusMode && ttsChunks.length > 0 ? (
        <TtsBar chunks={ttsChunks} onActiveChange={setActiveTtsId} />
      ) : null}
    </div>
  )
}
