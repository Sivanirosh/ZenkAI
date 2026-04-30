// Generic SWR-style hook.
//
// Contract:
//   - Returns `data` synchronously if the caller supplied a cached value
//     (rendered immediately on mount / tab switch).
//   - Kicks off `fetcher` in the background on every key change. The fetch
//     is given an AbortSignal that is fired on unmount or key change.
//   - If the fetch succeeds, `onData` is invoked so the caller can persist
//     into the global cache.
//   - AbortErrors are silent. `TypeError`s bubble up as `error` after the
//     one-shot retry in api.ts, so screens can show a retry banner.

'use client'

import { useEffect, useRef, useState } from 'react'
import { isAbort } from '@/lib/api'

export interface UseCachedResourceArgs<T> {
  /** Stable key for the resource. When it changes, refetch. */
  key: string | null
  /** Value from the global cache (rendered immediately if present). */
  cached: T | null | undefined
  /** Function that fetches fresh data. Must honour the provided signal. */
  fetcher: (signal: AbortSignal) => Promise<T>
  /** Called with the fresh result so the caller can update the store. */
  onData: (data: T) => void
}

export interface UseCachedResourceResult<T> {
  data: T | null
  loading: boolean
  error: string | null
  /** Manually re-run the fetcher (e.g. from a retry button). */
  refresh: () => void
}

export function useCachedResource<T>({
  key,
  cached,
  fetcher,
  onData,
}: UseCachedResourceArgs<T>): UseCachedResourceResult<T> {
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState<boolean>(!cached && key !== null)
  const [bump, setBump] = useState(0)
  // Keep the latest callbacks so the effect doesn't restart on every render.
  const fetcherRef = useRef(fetcher)
  const onDataRef = useRef(onData)
  fetcherRef.current = fetcher
  onDataRef.current = onData

  useEffect(() => {
    if (key === null) {
      setLoading(false)
      return
    }
    const ac = new AbortController()
    setError(null)
    if (!cached) setLoading(true)
    fetcherRef
      .current(ac.signal)
      .then((fresh) => {
        if (ac.signal.aborted) return
        onDataRef.current(fresh)
        setLoading(false)
      })
      .catch((err: unknown) => {
        if (isAbort(err) || ac.signal.aborted) return
        setError(err instanceof Error ? err.message : String(err))
        setLoading(false)
      })
    return () => ac.abort()
    // `cached` is intentionally excluded: we want to revalidate on key change
    // and on explicit refresh only, not every time the cache updates.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, bump])

  return {
    data: cached ?? null,
    loading,
    error,
    refresh: () => setBump((n) => n + 1),
  }
}
