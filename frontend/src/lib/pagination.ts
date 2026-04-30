// Pure client-side pagination helper: packs ordered paragraphs into
// reader-sized "pages" of roughly `targetWords` words, bounded by a
// minimum and maximum number of paragraphs per page.
//
// Used by the Leseraum so the reader sees one screen-worth of prose at
// a time with a clear "Seite X / Y" counter. Backend stays paragraph-
// oriented; pages are a purely presentational grouping.

export interface PaginationInput {
  id: string
  word_count: number
}

export interface Page {
  index: number
  paragraphIds: string[]
  wordCount: number
}

export interface PaginationOptions {
  targetWords: number
  minParas: number
  maxParas: number
}

const DEFAULTS: PaginationOptions = {
  targetWords: 350,
  minParas: 3,
  maxParas: 10,
}

export function buildPages(
  paragraphs: readonly PaginationInput[],
  opts: Partial<PaginationOptions> = {},
): Page[] {
  const { targetWords, minParas, maxParas } = { ...DEFAULTS, ...opts }
  if (paragraphs.length === 0) return []

  const pages: Page[] = []
  let currentIds: string[] = []
  let currentWords = 0

  const flush = () => {
    if (currentIds.length === 0) return
    pages.push({
      index: pages.length,
      paragraphIds: currentIds,
      wordCount: currentWords,
    })
    currentIds = []
    currentWords = 0
  }

  for (const p of paragraphs) {
    currentIds.push(p.id)
    currentWords += p.word_count ?? 0
    const hitTarget = currentWords >= targetWords
    const hitMax = currentIds.length >= maxParas
    if (hitTarget || hitMax) flush()
  }

  flush()

  // Merge an undersized tail into the previous page so we never end on a
  // stub of 1-2 paragraphs.
  if (pages.length >= 2) {
    const tail = pages[pages.length - 1]
    if (tail.paragraphIds.length < minParas) {
      const prev = pages[pages.length - 2]
      const merged: Page = {
        index: prev.index,
        paragraphIds: [...prev.paragraphIds, ...tail.paragraphIds],
        wordCount: prev.wordCount + tail.wordCount,
      }
      pages.splice(pages.length - 2, 2, merged)
    }
  }

  return pages.map((p, i) => ({ ...p, index: i }))
}

export function findPageIndex(pages: Page[], paragraphId: string): number {
  for (const page of pages) {
    if (page.paragraphIds.includes(paragraphId)) return page.index
  }
  return -1
}
