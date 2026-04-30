// SSE chat consumer scoped to the current paragraph.
//
// Owns the live transcript (the draft AI message being streamed) and
// appends completed turns to the session store keyed by paragraph id. The
// history shipped to the backend is the committed conversation; the
// in-flight draft is client-side only.

'use client'

import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { streamChat } from '@/lib/api'
import type { ChatTurn } from '@/lib/types'
import { useSession } from '@/store/session'

export type ChatStatus = 'idle' | 'streaming' | 'done' | 'error'

export interface UseChatResult {
  history: ChatTurn[]
  status: ChatStatus
  draft: string
  error: string | null
  ask: (question: string) => Promise<string>
  clear: () => void
}

export function useChat(paragraphId: string | null): UseChatResult {
  const chatByParagraph = useSession((s) => s.chatByParagraph)
  const appendChatTurn = useSession((s) => s.appendChatTurn)
  const clearChat = useSession((s) => s.clearChat)

  const [status, setStatus] = useState<ChatStatus>('idle')
  const [draft, setDraft] = useState('')
  const [error, setError] = useState<string | null>(null)

  const abortRef = useRef<AbortController | null>(null)

  const history: ChatTurn[] = useMemo(
    () => (paragraphId ? chatByParagraph[paragraphId] ?? [] : []),
    [chatByParagraph, paragraphId],
  )

  const clear = useCallback(() => {
    if (paragraphId) clearChat(paragraphId)
    setDraft('')
    setError(null)
    setStatus('idle')
  }, [paragraphId, clearChat])

  const ask = useCallback(
    async (question: string) => {
      const q = question.trim()
      if (!q) return ''
      const anchorId = paragraphId
      setStatus('streaming')
      setDraft('')
      setError(null)

      if (anchorId) {
        appendChatTurn(anchorId, { role: 'user', content: q })
      }

      if (abortRef.current) abortRef.current.abort()
      const ac = new AbortController()
      abortRef.current = ac

      let accumulated = ''
      await streamChat(
        {
          question: q,
          paragraph_id: anchorId,
          history,
        },
        {
          onToken: (delta) => {
            accumulated += delta
            setDraft(accumulated)
          },
          onError: (err) => {
            setError(err instanceof Error ? err.message : String(err))
            setStatus('error')
          },
          onDone: () => {
            if (anchorId && accumulated.trim().length > 0) {
              appendChatTurn(anchorId, {
                role: 'assistant',
                content: accumulated,
              })
            }
            setDraft('')
            setStatus((s) => (s === 'error' ? 'error' : 'done'))
          },
          signal: ac.signal,
        },
      )
      return accumulated
    },
    [paragraphId, history, appendChatTurn],
  )

  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort()
    }
  }, [])

  return { history, status, draft, error, ask, clear }
}
