// One clickable word in the reader: coloured by vocab status, click routes
// to the triage gestures (decision 0001). Punctuation and whitespace are
// never WordSpans — the Reader emits them as raw text between adjacent
// word ranges.
//
// Gestures:
//   - Plain click      → enqueue   (status 'new' → 'queued')
//   - Shift+click      → markKnown (status 'new' → 'known')
//   - Alt/Meta+click   → open Word Card (informational)
//   - Long-press       → equivalent to shift+click (touch parity)
//
// Words that are already 'known' render without decoration and have no
// click handler. Words that are 'queued' or 'exported' show the triage
// status and clicking them re-opens the Word Card for edit/removal.

'use client'

import { useRef } from 'react'
import type { WordToken } from '@/lib/types'
import { useSession } from '@/store/session'
import { useVocab, type VocabTriageStatus } from '@/hooks/useVocab'

interface Props {
  token: WordToken
  sentence: string
  paragraphId: string
}

function classForVocabStatus(status: VocabTriageStatus): string {
  if (status === 'queued') return 'w-queued'
  if (status === 'exported') return 'w-exported'
  if (status === 'known') return 'w-known'
  return 'w-new'
}

const LONG_PRESS_MS = 450

export function WordSpan({ token, sentence, paragraphId }: Props) {
  const { getStatus, enqueue, markKnown } = useVocab()
  const openWord = useSession((s) => s.openWord)

  const status: VocabTriageStatus = token.word_id
    ? getStatus(token.word_id)
    : 'known'

  const className = classForVocabStatus(status)

  const longPressTimer = useRef<number | null>(null)
  const longPressFired = useRef(false)

  const openCard = () => {
    if (!token.word_id) return
    openWord({
      wordId: token.word_id,
      surfaceForm: token.surface_form,
      sentence,
      grammaticalRole: token.grammatical_role,
      caseLabel: token.case_label,
    })
  }

  const handleClick = (event: React.MouseEvent<HTMLSpanElement>) => {
    if (longPressFired.current) {
      longPressFired.current = false
      return
    }
    if (!token.word_id) return

    // Alt or Meta → informational view (never mutates state).
    if (event.altKey || event.metaKey) {
      openCard()
      return
    }

    // Shift → mark as known.
    if (event.shiftKey) {
      void markKnown(token.word_id)
      return
    }

    // Already-triaged words: open card for edit/removal instead of re-queuing.
    if (status === 'queued' || status === 'exported') {
      openCard()
      return
    }
    if (status === 'known') {
      return
    }

    void enqueue(token.word_id, paragraphId, sentence)
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLSpanElement>) => {
    if (event.key !== 'Enter' && event.key !== ' ') return
    event.preventDefault()
    if (!token.word_id) return
    if (event.shiftKey) {
      void markKnown(token.word_id)
    } else if (event.altKey) {
      openCard()
    } else if (status === 'known') {
      return
    } else if (status === 'queued' || status === 'exported') {
      openCard()
    } else {
      void enqueue(token.word_id, paragraphId, sentence)
    }
  }

  const startLongPress = () => {
    if (!token.word_id) return
    longPressFired.current = false
    longPressTimer.current = window.setTimeout(() => {
      longPressFired.current = true
      if (token.word_id) void markKnown(token.word_id)
    }, LONG_PRESS_MS)
  }

  const cancelLongPress = () => {
    if (longPressTimer.current !== null) {
      window.clearTimeout(longPressTimer.current)
      longPressTimer.current = null
    }
  }

  const interactive = status !== 'known'

  return (
    <span
      className={className}
      onClick={interactive ? handleClick : undefined}
      onKeyDown={interactive ? handleKeyDown : undefined}
      onTouchStart={interactive ? startLongPress : undefined}
      onTouchEnd={interactive ? cancelLongPress : undefined}
      onTouchCancel={interactive ? cancelLongPress : undefined}
      onTouchMove={interactive ? cancelLongPress : undefined}
      role={interactive ? 'button' : undefined}
      tabIndex={interactive ? 0 : -1}
      data-vocab-status={status}
    >
      {token.surface_form}
    </span>
  )
}
