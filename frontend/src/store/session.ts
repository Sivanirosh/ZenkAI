// Zustand store: navigation state + in-memory data cache.
//
// Navigation + reader prefs (resume position, font scale, focus mode,
// scroll offset per page) are persisted to localStorage so re-opening the
// app returns to the exact same page at the exact same scroll position.
// The SWR caches for works/chapters/paragraphs/words/annotations stay
// session-scoped so ingestion changes can't serve stale data.

import { create } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'
import type {
  Chapter,
  ChatTurn,
  Paragraph,
  ParagraphWithTokens,
  ProgressSummary,
  TabId,
  VocabStatus,
  WordAnnotation,
  WordWithState,
  WorkWithProgress,
} from '@/lib/types'

export interface SelectedWord {
  wordId: string | null
  surfaceForm: string
  sentence: string
  grammaticalRole: string | null
  caseLabel: string | null
}

export interface LibraryCache {
  works: WorkWithProgress[]
  summary: ProgressSummary
  fetchedAt: number
}

export interface ReaderPrefs {
  fontScale: number
  focusMode: boolean
  ttsSpeed: 0.8 | 1.0 | 1.2
  ttsAutoPlay: boolean
}

interface SessionState {
  activeTab: TabId
  currentWorkId: string | null
  currentParagraphId: string | null
  epochFilter: string | null
  selectedWord: SelectedWord | null

  wordFamiliarity: Record<string, number>

  // Vocab harvester status map (decision 0001). Absence = 'new'.
  vocabStatus: Record<string, VocabStatus>

  lastParagraphByWork: Record<string, string>
  readerPrefs: ReaderPrefs
  lastScrollByPage: Record<string, number>

  // Volatile (not persisted): live chat history keyed by paragraph id so the
  // conversation resets when the reader moves on.
  chatByParagraph: Record<string, ChatTurn[]>

  libraryCache: LibraryCache | null
  chaptersCache: Record<string, Chapter[]>
  // Full paragraph metadata per chapter (id + word_count), used to compute
  // page bundles in the Leseraum. Keyed by `${workId}:${chapter}`.
  chapterParagraphsCache: Record<string, Paragraph[]>
  paragraphCache: Record<string, ParagraphWithTokens>
  wordCache: Record<string, WordWithState>
  annotationCache: Record<string, WordAnnotation>

  setTab: (tab: TabId) => void
  openWork: (workId: string, firstParagraphId: string | null) => void
  setCurrentParagraph: (paragraphId: string) => void
  setEpochFilter: (epoch: string | null) => void
  openWord: (word: SelectedWord) => void
  clearSelectedWord: () => void
  setWordFamiliarity: (wordId: string, familiarity: number) => void
  bulkSetWordFamiliarity: (entries: Record<string, number>) => void

  setVocabStatus: (wordId: string, status: VocabStatus | null) => void
  bulkSetVocabStatus: (entries: Record<string, VocabStatus>) => void

  setReaderPrefs: (patch: Partial<ReaderPrefs>) => void
  setPageScroll: (key: string, top: number) => void

  appendChatTurn: (paragraphId: string, turn: ChatTurn) => void
  updateLastChatTurn: (paragraphId: string, content: string) => void
  clearChat: (paragraphId: string) => void

  setLibraryCache: (cache: LibraryCache) => void
  setChaptersCache: (workId: string, chapters: Chapter[]) => void
  setChapterParagraphsCache: (
    workId: string,
    chapter: number,
    paragraphs: Paragraph[],
  ) => void
  cacheParagraph: (paragraph: ParagraphWithTokens) => void
  cacheWord: (word: WordWithState) => void
  cacheAnnotation: (key: string, annotation: WordAnnotation) => void
}

export const chapterKey = (workId: string, chapter: number): string =>
  `${workId}:${chapter}`

export const pageScrollKey = (
  workId: string,
  chapter: number,
  pageIdx: number,
): string => `${workId}:${chapter}:${pageIdx}`

const DEFAULT_READER_PREFS: ReaderPrefs = {
  fontScale: 1,
  focusMode: false,
  ttsSpeed: 1.0,
  ttsAutoPlay: true,
}

export const useSession = create<SessionState>()(
  persist(
    (set) => ({
      activeTab: 'lib',
      currentWorkId: null,
      currentParagraphId: null,
      epochFilter: null,
      selectedWord: null,
      wordFamiliarity: {},
      vocabStatus: {},

      lastParagraphByWork: {},
      readerPrefs: DEFAULT_READER_PREFS,
      lastScrollByPage: {},

      chatByParagraph: {},

      libraryCache: null,
      chaptersCache: {},
      chapterParagraphsCache: {},
      paragraphCache: {},
      wordCache: {},
      annotationCache: {},

      setTab: (tab) => set({ activeTab: tab }),
      openWork: (workId, firstParagraphId) =>
        set((state) => {
          const remembered = state.lastParagraphByWork[workId] ?? null
          return {
            activeTab: 'rd',
            currentWorkId: workId,
            currentParagraphId: remembered ?? firstParagraphId,
          }
        }),
      setCurrentParagraph: (paragraphId) =>
        set((state) => {
          const patch: Partial<SessionState> = {
            currentParagraphId: paragraphId,
          }
          if (state.currentWorkId) {
            patch.lastParagraphByWork = {
              ...state.lastParagraphByWork,
              [state.currentWorkId]: paragraphId,
            }
          }
          return patch as SessionState
        }),
      setEpochFilter: (epoch) => set({ epochFilter: epoch }),
      openWord: (word) => set({ selectedWord: word, activeTab: 'wc' }),
      clearSelectedWord: () => set({ selectedWord: null }),

      setWordFamiliarity: (wordId, familiarity) =>
        set((state) => ({
          wordFamiliarity: {
            ...state.wordFamiliarity,
            [wordId]: Math.max(state.wordFamiliarity[wordId] ?? 0, familiarity),
          },
        })),
      bulkSetWordFamiliarity: (entries) =>
        set((state) => {
          const next = { ...state.wordFamiliarity }
          for (const [id, fam] of Object.entries(entries)) {
            next[id] = Math.max(next[id] ?? 0, fam)
          }
          return { wordFamiliarity: next }
        }),

      setVocabStatus: (wordId, status) =>
        set((state) => {
          const next = { ...state.vocabStatus }
          if (status === null) delete next[wordId]
          else next[wordId] = status
          return { vocabStatus: next }
        }),
      bulkSetVocabStatus: (entries) =>
        set((state) => ({
          vocabStatus: { ...state.vocabStatus, ...entries },
        })),

      setReaderPrefs: (patch) =>
        set((state) => ({
          readerPrefs: { ...state.readerPrefs, ...patch },
        })),
      setPageScroll: (key, top) =>
        set((state) => ({
          lastScrollByPage: { ...state.lastScrollByPage, [key]: top },
        })),

      appendChatTurn: (paragraphId, turn) =>
        set((state) => {
          const existing = state.chatByParagraph[paragraphId] ?? []
          return {
            chatByParagraph: {
              ...state.chatByParagraph,
              [paragraphId]: [...existing, turn],
            },
          }
        }),
      updateLastChatTurn: (paragraphId, content) =>
        set((state) => {
          const existing = state.chatByParagraph[paragraphId] ?? []
          if (existing.length === 0) return state
          const next = existing.slice(0, -1)
          next.push({ ...existing[existing.length - 1], content })
          return {
            chatByParagraph: {
              ...state.chatByParagraph,
              [paragraphId]: next,
            },
          }
        }),
      clearChat: (paragraphId) =>
        set((state) => {
          const next = { ...state.chatByParagraph }
          delete next[paragraphId]
          return { chatByParagraph: next }
        }),

      setLibraryCache: (cache) => set({ libraryCache: cache }),
      setChaptersCache: (workId, chapters) =>
        set((state) => ({
          chaptersCache: { ...state.chaptersCache, [workId]: chapters },
        })),
      setChapterParagraphsCache: (workId, chapter, paragraphs) =>
        set((state) => ({
          chapterParagraphsCache: {
            ...state.chapterParagraphsCache,
            [chapterKey(workId, chapter)]: paragraphs,
          },
        })),
      cacheParagraph: (paragraph) =>
        set((state) => ({
          paragraphCache: { ...state.paragraphCache, [paragraph.id]: paragraph },
        })),
      cacheWord: (word) =>
        set((state) => ({
          wordCache: { ...state.wordCache, [word.id]: word },
        })),
      cacheAnnotation: (key, annotation) =>
        set((state) => ({
          annotationCache: { ...state.annotationCache, [key]: annotation },
        })),
    }),
    {
      name: 'zenkai-session',
      storage: createJSONStorage(() => localStorage),
      version: 2,
      // Persist navigation + reader prefs only. SWR caches stay volatile.
      partialize: (state) => ({
        activeTab: state.activeTab,
        currentWorkId: state.currentWorkId,
        currentParagraphId: state.currentParagraphId,
        epochFilter: state.epochFilter,
        lastParagraphByWork: state.lastParagraphByWork,
        readerPrefs: state.readerPrefs,
        lastScrollByPage: state.lastScrollByPage,
        wordFamiliarity: state.wordFamiliarity,
        vocabStatus: state.vocabStatus,
      }),
      migrate: (persisted: unknown, version: number) => {
        if (
          persisted &&
          typeof persisted === 'object' &&
          'readerPrefs' in persisted &&
          version < 2
        ) {
          const prev = (persisted as { readerPrefs: Partial<ReaderPrefs> })
            .readerPrefs
          return {
            ...(persisted as object),
            readerPrefs: { ...DEFAULT_READER_PREFS, ...prev },
          }
        }
        return persisted as Partial<SessionState>
      },
    },
  ),
)
