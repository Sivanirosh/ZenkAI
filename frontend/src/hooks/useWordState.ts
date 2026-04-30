// DEPRECATED — decision 0001. Prefer `useVocab`. This hook is only kept
// for the informational read-only familiarity display inside the Word Card;
// all other UI (Reader gestures, Export actions) has moved to `useVocab`.
//
// The backend endpoints this talks to are marked `deprecated=True` and may
// be removed in the next migration cycle.

'use client'

import { useCallback } from 'react'
import * as api from '@/lib/api'
import { useSession } from '@/store/session'

export function useWordState() {
  const familiarityMap = useSession((s) => s.wordFamiliarity)
  const setFamiliarity = useSession((s) => s.setWordFamiliarity)
  const bulkSet = useSession((s) => s.bulkSetWordFamiliarity)

  const getFamiliarity = useCallback(
    (wordId: string, fallback = 0): number => familiarityMap[wordId] ?? fallback,
    [familiarityMap],
  )

  const markSeen = useCallback(
    async (wordId: string) => {
      const current = familiarityMap[wordId] ?? 0
      if (current < 1) setFamiliarity(wordId, 1)
      try {
        const state = await api.markSeen(wordId)
        setFamiliarity(wordId, state.familiarity)
      } catch {
        // network hiccup — keep the optimistic update; next refresh will correct it.
      }
    },
    [familiarityMap, setFamiliarity],
  )

  const markOpened = useCallback(
    async (wordId: string) => {
      const current = familiarityMap[wordId] ?? 0
      if (current < 2) setFamiliarity(wordId, 2)
      try {
        const state = await api.markOpened(wordId)
        setFamiliarity(wordId, state.familiarity)
      } catch {
        // swallow — optimistic value stays
      }
    },
    [familiarityMap, setFamiliarity],
  )

  const markKnown = useCallback(
    async (wordId: string) => {
      setFamiliarity(wordId, 4)
      try {
        const state = await api.markKnown(wordId)
        setFamiliarity(wordId, state.familiarity)
      } catch {
        // swallow
      }
    },
    [setFamiliarity],
  )

  const markReviewed = useCallback(
    async (wordId: string, quality: number) => {
      try {
        const state = await api.markReviewed(wordId, quality)
        setFamiliarity(wordId, state.familiarity)
      } catch {
        // swallow
      }
    },
    [setFamiliarity],
  )

  return {
    getFamiliarity,
    markSeen,
    markOpened,
    markKnown,
    markReviewed,
    bulkSet,
  }
}
