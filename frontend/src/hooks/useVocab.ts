// Vocabulary harvester hook — the reader's state handle post-decision 0001.
//
// Replaces `useWordState` for reader interactions. The old hook is kept for
// the informational Word Card only (read-only familiarity display).

'use client'

import { useCallback } from 'react'
import * as api from '@/lib/api'
import type { VocabStatus } from '@/lib/types'
import { useSession } from '@/store/session'

export type VocabTriageStatus = 'new' | VocabStatus

export function useVocab() {
  const statusMap = useSession((s) => s.vocabStatus)
  const setStatus = useSession((s) => s.setVocabStatus)
  const bulkSet = useSession((s) => s.bulkSetVocabStatus)

  const getStatus = useCallback(
    (wordId: string): VocabTriageStatus => statusMap[wordId] ?? 'new',
    [statusMap],
  )

  const enqueue = useCallback(
    async (
      wordId: string,
      paragraphId: string | null,
      sentence: string | null,
    ) => {
      const prev = statusMap[wordId] ?? null
      // Optimistic: exported rows stay exported until server confirms re-queue.
      if (prev !== 'exported') setStatus(wordId, 'queued')
      try {
        const entry = await api.vocabEnqueue(wordId, paragraphId, sentence)
        setStatus(wordId, entry.status)
      } catch (err) {
        // Roll back optimistic state if the server rejected it.
        setStatus(wordId, prev)
        throw err
      }
    },
    [statusMap, setStatus],
  )

  const markKnown = useCallback(
    async (wordId: string) => {
      const prev = statusMap[wordId] ?? null
      setStatus(wordId, 'known')
      try {
        const entry = await api.vocabMarkKnown(wordId)
        setStatus(wordId, entry.status)
      } catch (err) {
        setStatus(wordId, prev)
        throw err
      }
    },
    [statusMap, setStatus],
  )

  const remove = useCallback(
    async (wordId: string) => {
      const prev = statusMap[wordId] ?? null
      setStatus(wordId, null)
      try {
        await api.vocabRemove(wordId)
      } catch (err) {
        setStatus(wordId, prev)
        throw err
      }
    },
    [statusMap, setStatus],
  )

  return {
    getStatus,
    enqueue,
    markKnown,
    remove,
    bulkSet,
    statusMap,
  }
}
