'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import { streamAgentTurn } from '../lib/api'
import type {
  MiraEvent,
  MiraTurnRequest,
  ToolError,
  ToolIntent,
  ToolResult,
} from '../lib/types'

export type MiraStatus = 'idle' | 'streaming' | 'done' | 'error' | 'cancelled'

export interface UseMiraResult {
  thoughts: string[]
  intents: ToolIntent[]
  results: ToolResult[]
  errors: ToolError[]
  status: MiraStatus
  sessionId: string | null
  /** Send a turn request and stream events. Cancels any in-flight stream first. */
  run: (payload: MiraTurnRequest) => Promise<void>
  /** Abort the current SSE stream (no-op if idle). */
  cancel: () => void
  /** Reset the local buffers (e.g. before re-rendering a fresh turn). */
  reset: () => void
}

/**
 * useMira — owns one in-flight Mira turn and the live event buffers
 * the UI binds to.
 *
 * Contract:
 * - Calling `run` while a previous turn is streaming aborts the
 *   previous stream and immediately starts the new one.
 * - The `status` stays `'streaming'` until the backend emits a `done`
 *   event (or the user calls `cancel`). After that, the buffers stay
 *   populated so the UI can keep rendering the last turn.
 * - The hook cleans up the AbortController on unmount.
 */
export function useMira(): UseMiraResult {
  const [thoughts, setThoughts] = useState<string[]>([])
  const [intents, setIntents] = useState<ToolIntent[]>([])
  const [results, setResults] = useState<ToolResult[]>([])
  const [errors, setErrors] = useState<ToolError[]>([])
  const [status, setStatus] = useState<MiraStatus>('idle')
  const [sessionId, setSessionId] = useState<string | null>(null)

  const abortRef = useRef<AbortController | null>(null)

  const reset = useCallback(() => {
    setThoughts([])
    setIntents([])
    setResults([])
    setErrors([])
    setStatus('idle')
    setSessionId(null)
  }, [])

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
      setStatus('cancelled')
    }
  }, [])

  const run = useCallback(async (payload: MiraTurnRequest) => {
    if (abortRef.current) {
      abortRef.current.abort()
    }
    const controller = new AbortController()
    abortRef.current = controller

    setThoughts([])
    setIntents([])
    setResults([])
    setErrors([])
    setStatus('streaming')

    try {
      for await (const ev of streamAgentTurn(payload, { signal: controller.signal })) {
        applyEvent(ev, {
          setThoughts,
          setIntents,
          setResults,
          setErrors,
          setStatus,
          setSessionId,
        })
      }
    } catch (err) {
      if ((err as { name?: string }).name === 'AbortError') return
      setErrors((prev) => [
        ...prev,
        { tool: 'mira', code: 'stream_error', message: String(err) },
      ])
      setStatus('error')
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    return () => {
      abortRef.current?.abort()
      abortRef.current = null
    }
  }, [])

  return {
    thoughts,
    intents,
    results,
    errors,
    status,
    sessionId,
    run,
    cancel,
    reset,
  }
}

interface SetterBag {
  setThoughts: React.Dispatch<React.SetStateAction<string[]>>
  setIntents: React.Dispatch<React.SetStateAction<ToolIntent[]>>
  setResults: React.Dispatch<React.SetStateAction<ToolResult[]>>
  setErrors: React.Dispatch<React.SetStateAction<ToolError[]>>
  setStatus: React.Dispatch<React.SetStateAction<MiraStatus>>
  setSessionId: React.Dispatch<React.SetStateAction<string | null>>
}

function applyEvent(ev: MiraEvent, s: SetterBag): void {
  switch (ev.kind) {
    case 'thought':
      s.setThoughts((prev) => [...prev, ev.data])
      return
    case 'tool_intent':
      s.setIntents((prev) => [...prev, ev.data])
      return
    case 'tool_result':
      s.setResults((prev) => [...prev, ev.data])
      return
    case 'tool_error':
      s.setErrors((prev) => [...prev, ev.data])
      return
    case 'error':
      s.setErrors((prev) => [
        ...prev,
        {
          tool: 'mira',
          code: typeof ev.data?.message === 'string' ? ev.data.message : 'error',
          message: typeof ev.data?.message === 'string' ? ev.data.message : 'error',
          detail: ev.data?.detail,
        } as ToolError,
      ])
      s.setStatus('error')
      return
    case 'done':
      if (ev.data?.session_id) s.setSessionId(ev.data.session_id)
      s.setStatus((prev) => (prev === 'error' ? 'error' : 'done'))
      return
  }
}
