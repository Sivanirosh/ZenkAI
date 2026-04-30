// Typed fetch wrappers for the ZenkAI backend.
// Every call to `/api/v1/*` must go through this module (CLAUDE.md rule).

import type {
  AtlasPayload,
  AtriumPayload,
  CaptureRecentResponse,
  CaptureResultPayload,
  CaptureSurfaceKind,
  Chapter,
  ChatRequest,
  KonversationRecentResponse,
  KonversationTurnResponse,
  MiraEvent,
  MiraTurnRequest,
  OnboardingGoalResponse,
  OnboardingState,
  Paragraph,
  ParagraphWithTokens,
  ProgressSummary,
  PronunciationScore,
  ServiceUnavailable,
  SttResponse,
  ToolError,
  ToolIntent,
  ToolResult,
  VocabCounts,
  VocabEntry,
  VocabStatus,
  VocabStatusMap,
  WordAnnotation,
  WordState,
  WordWithState,
  WorkWithProgress,
} from './types'

const BASE = '/api/v1'

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export interface ApiOptions {
  signal?: AbortSignal
}

function isAbort(err: unknown): boolean {
  return (
    err instanceof DOMException && err.name === 'AbortError'
  ) || (err instanceof Error && err.name === 'AbortError')
}

async function request<T>(
  path: string,
  init: RequestInit & ApiOptions = {},
): Promise<T> {
  const method = (init.method ?? 'GET').toUpperCase()
  const doFetch = () =>
    fetch(`${BASE}${path}`, {
      headers: {
        'Content-Type': 'application/json',
        ...(init.headers ?? {}),
      },
      ...init,
    })

  let response: Response
  try {
    response = await doFetch()
  } catch (err) {
    // `TypeError: Failed to fetch` surfaces when a pooled keep-alive
    // connection in the Next.js dev rewrite layer was already closed by
    // uvicorn. For idempotent GETs we retry once with a fresh socket.
    if (isAbort(err)) throw err
    if (method === 'GET' && err instanceof TypeError) {
      response = await doFetch()
    } else {
      throw err
    }
  }

  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new ApiError(
      `${response.status} ${response.statusText}${text ? ` \u2014 ${text}` : ''}`,
      response.status,
    )
  }
  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

// ── Corpus ─────────────────────────────────────────────

export const listWorks = (opts?: ApiOptions): Promise<WorkWithProgress[]> =>
  request<WorkWithProgress[]>('/corpus/works', opts)

export const listChapters = (
  workId: string,
  opts?: ApiOptions,
): Promise<Chapter[]> =>
  request<Chapter[]>(
    `/corpus/works/${encodeURIComponent(workId)}/chapters`,
    opts,
  )

export const listChapterParagraphs = (
  workId: string,
  chapter: number,
  opts?: ApiOptions,
): Promise<Paragraph[]> =>
  request<Paragraph[]>(
    `/corpus/works/${encodeURIComponent(workId)}/chapters/${chapter}/paragraphs`,
    opts,
  )

export const getParagraph = (
  paragraphId: string,
  opts?: ApiOptions,
): Promise<ParagraphWithTokens> =>
  request<ParagraphWithTokens>(
    `/corpus/paragraphs/${encodeURIComponent(paragraphId)}`,
    opts,
  )

// Must match ParagraphBatchRequest.ids max_length on the backend.
export const PARAGRAPH_BATCH_LIMIT = 32

export const getParagraphs = (
  ids: string[],
  opts?: ApiOptions,
): Promise<ParagraphWithTokens[]> => {
  if (ids.length === 0) return Promise.resolve([])
  if (ids.length > PARAGRAPH_BATCH_LIMIT) {
    return Promise.reject(
      new Error(
        `getParagraphs: batch size ${ids.length} exceeds limit ${PARAGRAPH_BATCH_LIMIT}`,
      ),
    )
  }
  return request<ParagraphWithTokens[]>('/corpus/paragraphs/batch', {
    method: 'POST',
    body: JSON.stringify({ ids }),
    ...opts,
  })
}

// ── Words (legacy SRS — deprecated, see decision 0001) ─────────

export const getWord = (
  wordId: string,
  opts?: ApiOptions,
): Promise<WordWithState> =>
  request<WordWithState>(`/words/${encodeURIComponent(wordId)}`, opts)

/** @deprecated External SRS owns scheduling. Use vocab APIs. */
export const markSeen = (wordId: string): Promise<WordState> =>
  request<WordState>(`/words/${encodeURIComponent(wordId)}/seen`, {
    method: 'POST',
  })

/** @deprecated External SRS owns scheduling. Use vocab APIs. */
export const markOpened = (wordId: string): Promise<WordState> =>
  request<WordState>(`/words/${encodeURIComponent(wordId)}/opened`, {
    method: 'POST',
  })

/** @deprecated External SRS owns scheduling. Use vocab APIs. */
export const markReviewed = (
  wordId: string,
  quality: number,
): Promise<WordState> =>
  request<WordState>(`/words/${encodeURIComponent(wordId)}/reviewed`, {
    method: 'POST',
    body: JSON.stringify({ quality }),
  })

/** @deprecated External SRS owns scheduling. Use `vocabMarkKnown`. */
export const markKnown = (wordId: string): Promise<WordState> =>
  request<WordState>(`/words/${encodeURIComponent(wordId)}/known`, {
    method: 'POST',
  })

/** @deprecated External SRS owns scheduling. */
export const getReviewQueue = (
  limit = 20,
  opts?: ApiOptions,
): Promise<WordState[]> =>
  request<WordState[]>(`/words/queue/review?limit=${limit}`, opts)

export const getProgress = (opts?: ApiOptions): Promise<ProgressSummary> =>
  request<ProgressSummary>('/words/progress', opts)

// ── Vocabulary harvester ───────────────────────────────

export const vocabEnqueue = (
  wordId: string,
  paragraphId: string | null,
  sentence: string | null,
): Promise<VocabEntry> =>
  request<VocabEntry>('/vocab/queue', {
    method: 'POST',
    body: JSON.stringify({
      word_id: wordId,
      paragraph_id: paragraphId,
      sentence,
    }),
  })

export const vocabMarkKnown = (wordId: string): Promise<VocabEntry> =>
  request<VocabEntry>('/vocab/known', {
    method: 'POST',
    body: JSON.stringify({ word_id: wordId }),
  })

export const vocabRemove = (wordId: string): Promise<void> =>
  request<void>(`/vocab/queue/${encodeURIComponent(wordId)}`, {
    method: 'DELETE',
  })

export const vocabList = (
  status: VocabStatus | 'all' = 'queued',
  opts?: ApiOptions,
): Promise<VocabEntry[]> =>
  request<VocabEntry[]>(`/vocab/queue?status=${encodeURIComponent(status)}`, opts)

export const vocabCounts = (opts?: ApiOptions): Promise<VocabCounts> =>
  request<VocabCounts>('/vocab/counts', opts)

export const vocabStatusMap = (
  wordIds: string[],
  opts?: ApiOptions,
): Promise<VocabStatusMap> =>
  request<VocabStatusMap>('/vocab/statuses', {
    method: 'POST',
    body: JSON.stringify({ word_ids: wordIds }),
    ...opts,
  })

export interface VocabOverrideBody {
  question?: string | null
  answer?: string | null
  extra_tags?: string | null
}

export const vocabUpdateOverrides = (
  wordId: string,
  body: VocabOverrideBody,
): Promise<VocabEntry> =>
  request<VocabEntry>(`/vocab/queue/${encodeURIComponent(wordId)}`, {
    method: 'PATCH',
    body: JSON.stringify(body),
  })

export const vocabAnnotate = (wordId: string): Promise<VocabEntry> =>
  request<VocabEntry>(
    `/vocab/queue/${encodeURIComponent(wordId)}/annotate`,
    { method: 'POST' },
  )

/** Triggers a CSV download. Returns true on success (file received), false
 *  if the server returned 204 (queue empty). */
export async function vocabExport(): Promise<boolean> {
  const response = await fetch(`${BASE}/vocab/export`, { method: 'POST' })
  if (response.status === 204) return false
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new ApiError(
      `${response.status} ${response.statusText}${text ? ` \u2014 ${text}` : ''}`,
      response.status,
    )
  }
  const blob = await response.blob()
  const disposition = response.headers.get('content-disposition') || ''
  const match = disposition.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i)
  const filename =
    match?.[1]?.trim() ||
    `linguamate-vocab-${new Date().toISOString().replace(/[:.]/g, '-')}.csv`
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 0)
  return true
}

// ── Annotations ────────────────────────────────────────

export interface WordAnnotationRequest {
  word: string
  sentence: string
  grammatical_role: string | null
  case_label: string | null
}

export const getWordAnnotation = (
  body: WordAnnotationRequest,
  opts?: ApiOptions,
): Promise<WordAnnotation> =>
  request<WordAnnotation>('/annotations/word', {
    method: 'POST',
    body: JSON.stringify(body),
    ...opts,
  })

// ── Chat (SSE) ─────────────────────────────────────────

export class ServiceUnavailableError extends ApiError {
  constructor(
    message: string,
    public info: ServiceUnavailable,
  ) {
    super(message, 503)
    this.name = 'ServiceUnavailableError'
  }
}

export interface StreamChatHandlers {
  onToken: (delta: string) => void
  onDone?: () => void
  onError?: (err: unknown) => void
  signal?: AbortSignal
}

export async function streamChat(
  body: ChatRequest,
  handlers: StreamChatHandlers,
): Promise<void> {
  const response = await fetch(`${BASE}/chat/message`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify(body),
    signal: handlers.signal,
  })

  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => '')
    handlers.onError?.(
      new ApiError(
        `${response.status} ${response.statusText}${text ? ` \u2014 ${text}` : ''}`,
        response.status,
      ),
    )
    handlers.onDone?.()
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const flushEvent = (block: string): boolean => {
    let event = 'message'
    const dataLines: string[] = []
    for (const line of block.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim()
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
    }
    if (dataLines.length === 0) return true
    const raw = dataLines.join('\n')
    if (event === 'token') {
      try {
        const obj = JSON.parse(raw) as { delta?: string }
        if (obj.delta) handlers.onToken(obj.delta)
      } catch (err) {
        handlers.onError?.(err)
      }
      return true
    }
    if (event === 'error') {
      try {
        const obj = JSON.parse(raw) as { detail?: string }
        handlers.onError?.(new Error(obj.detail || 'stream error'))
      } catch {
        handlers.onError?.(new Error(raw))
      }
      return true
    }
    if (event === 'done') return false
    return true
  }

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        const keepGoing = flushEvent(block)
        if (!keepGoing) {
          reader.cancel().catch(() => undefined)
          handlers.onDone?.()
          return
        }
      }
    }
    if (buffer.trim().length > 0) flushEvent(buffer)
  } catch (err) {
    if (!isAbort(err)) handlers.onError?.(err)
  } finally {
    handlers.onDone?.()
  }
}

// ── Mira agent (Atrium + SSE turn) ─────────────────────

export const getAtrium = (opts?: ApiOptions): Promise<AtriumPayload> =>
  request<AtriumPayload>('/agent/atrium', opts)

export const getAtlas = (opts?: ApiOptions): Promise<AtlasPayload> =>
  request<AtlasPayload>('/agent/atlas', opts)

// ── Onboarding (PIVOT_ROADMAP §B.1) ─────────────────────

export const getOnboardingState = (
  opts?: ApiOptions,
): Promise<OnboardingState> =>
  request<OnboardingState>('/onboarding/state', opts)

export const submitOnboardingGoal = (
  rawText: string,
  horizon: string = '30d',
): Promise<OnboardingGoalResponse> =>
  request<OnboardingGoalResponse>('/onboarding/goal', {
    method: 'POST',
    body: JSON.stringify({ raw_text: rawText, horizon }),
  })

// ── Konversation (PIVOT_ROADMAP §B.5/§B.17) ────────────

export interface KonversationTurnInput {
  audio: Blob
  sessionId: string
  competencyId?: string | null
  scenario?: string | null
  targetCefr?: string | null
  language?: string
}

export async function konversationTurn(
  input: KonversationTurnInput,
  opts: ApiOptions = {},
): Promise<KonversationTurnResponse> {
  const form = new FormData()
  const filename = input.audio.type.includes('webm')
    ? 'turn.webm'
    : input.audio.type.includes('ogg')
    ? 'turn.ogg'
    : 'turn.wav'
  form.append('audio', input.audio, filename)
  form.append('session_id', input.sessionId)
  if (input.competencyId) form.append('competency_id', input.competencyId)
  if (input.scenario) form.append('scenario', input.scenario)
  if (input.targetCefr) form.append('target_cefr', input.targetCefr)
  if (input.language) form.append('language', input.language)
  const response = await fetch(`${BASE}/conversation/turn`, {
    method: 'POST',
    body: form,
    signal: opts.signal,
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new ApiError(
      `${response.status} ${response.statusText}${text ? ` \u2014 ${text}` : ''}`,
      response.status,
    )
  }
  return (await response.json()) as KonversationTurnResponse
}

export const konversationRecent = (
  limit = 20,
  opts?: ApiOptions,
): Promise<KonversationRecentResponse> =>
  request<KonversationRecentResponse>(
    `/conversation/recent?limit=${limit}`,
    opts,
  )

// ── Konversation streaming (SSE) ────────────────────────

export interface KonversationStreamHandlers {
  /** Called as soon as STT finishes — show the user bubble immediately. */
  onTranscribed: (t: KonversationTurnResponse['user_transcript']) => void
  /** Each LLM token chunk — append to assistant bubble. */
  onToken: (delta: string) => void
  /** Stream is complete — full assistant text + metadata for TTS. */
  onDone: (result: KonversationTurnResponse & { duration_ms: number }) => void
  onError?: (err: unknown) => void
  signal?: AbortSignal
}

export async function konversationStream(
  input: KonversationTurnInput,
  handlers: KonversationStreamHandlers,
): Promise<void> {
  const form = new FormData()
  const filename = input.audio.type.includes('webm')
    ? 'turn.webm'
    : input.audio.type.includes('ogg')
    ? 'turn.ogg'
    : 'turn.wav'
  form.append('audio', input.audio, filename)
  form.append('session_id', input.sessionId)
  if (input.competencyId) form.append('competency_id', input.competencyId)
  if (input.scenario) form.append('scenario', input.scenario)
  if (input.targetCefr) form.append('target_cefr', input.targetCefr)
  if (input.language) form.append('language', input.language)

  let response: Response
  try {
    response = await fetch(`${BASE}/conversation/stream`, {
      method: 'POST',
      body: form,
      signal: handlers.signal,
    })
  } catch (err) {
    if (isAbort(err)) return
    handlers.onError?.(err)
    return
  }

  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => '')
    handlers.onError?.(
      new ApiError(
        `${response.status} ${response.statusText}${text ? ` — ${text}` : ''}`,
        response.status,
      ),
    )
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        let eventName = 'message'
        const dataLines: string[] = []
        for (const line of block.split('\n')) {
          if (line.startsWith('event:')) eventName = line.slice(6).trim()
          else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
        }
        if (dataLines.length === 0) continue
        const raw = dataLines.join('\n')
        try {
          const parsed = JSON.parse(raw)
          if (eventName === 'transcribed') {
            handlers.onTranscribed(parsed)
          } else if (eventName === 'token') {
            if (parsed.delta) handlers.onToken(parsed.delta as string)
          } else if (eventName === 'done') {
            reader.cancel().catch(() => undefined)
            handlers.onDone(parsed)
            return
          }
        } catch (err) {
          handlers.onError?.(err)
        }
      }
    }
  } catch (err) {
    if (!isAbort(err)) handlers.onError?.(err)
  }
}

// ── Pronunciation diff (PIVOT_ROADMAP §B.8) ───────────

export interface ScorePronunciationInput {
  audio: Blob
  referenceText: string
  targetCefr?: string | null
}

export async function scorePronunciation(
  input: ScorePronunciationInput,
  opts: ApiOptions = {},
): Promise<PronunciationScore> {
  const form = new FormData()
  const filename = input.audio.type.includes('webm')
    ? 'learner.webm'
    : input.audio.type.includes('ogg')
    ? 'learner.ogg'
    : 'learner.wav'
  form.append('audio', input.audio, filename)
  form.append('reference_text', input.referenceText)
  if (input.targetCefr) form.append('target_cefr', input.targetCefr)
  const response = await fetch(`${BASE}/conversation/pronounce`, {
    method: 'POST',
    body: form,
    signal: opts.signal,
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new ApiError(
      `${response.status} ${response.statusText}${text ? ` \u2014 ${text}` : ''}`,
      response.status,
    )
  }
  return (await response.json()) as PronunciationScore
}

// ── Capture (PIVOT_ROADMAP §B.6/§B.7/§B.18) ────────────

export interface CaptureImageInput {
  image: Blob
  sessionId: string
  surfaceKind?: CaptureSurfaceKind | string
  targetCompetencyId?: string | null
  targetCefr?: string | null
  note?: string | null
}

export async function captureImage(
  input: CaptureImageInput,
  opts: ApiOptions = {},
): Promise<CaptureResultPayload> {
  const form = new FormData()
  const filename = input.image.type.includes('webp')
    ? 'frame.webp'
    : input.image.type.includes('jpeg') || input.image.type.includes('jpg')
    ? 'frame.jpg'
    : 'frame.png'
  form.append('image', input.image, filename)
  form.append('session_id', input.sessionId)
  if (input.surfaceKind) form.append('surface_kind', input.surfaceKind)
  if (input.targetCompetencyId)
    form.append('target_competency_id', input.targetCompetencyId)
  if (input.targetCefr) form.append('target_cefr', input.targetCefr)
  if (input.note) form.append('note', input.note)
  const response = await fetch(`${BASE}/capture/image`, {
    method: 'POST',
    body: form,
    signal: opts.signal,
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new ApiError(
      `${response.status} ${response.statusText}${text ? ` \u2014 ${text}` : ''}`,
      response.status,
    )
  }
  return (await response.json()) as CaptureResultPayload
}

export const captureRecent = (
  limit = 20,
  opts?: ApiOptions,
): Promise<CaptureRecentResponse> =>
  request<CaptureRecentResponse>(`/capture/recent?limit=${limit}`, opts)

/**
 * Open a streaming POST to /agent/turn and yield typed MiraEvents.
 *
 * Cancel via `opts.signal`. The async iterator finishes naturally when
 * the backend emits a `done` event OR when the connection closes.
 *
 * Errors raised before the first byte (HTTP 4xx/5xx) become an
 * `{ kind: 'error' }` event followed by an `{ kind: 'done' }` so the
 * consumer never has to special-case the connection-failure branch.
 */
export async function* streamAgentTurn(
  payload: MiraTurnRequest,
  opts: ApiOptions = {},
): AsyncGenerator<MiraEvent, void, void> {
  let response: Response
  try {
    response = await fetch(`${BASE}/agent/turn`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(payload),
      signal: opts.signal,
    })
  } catch (err) {
    if (isAbort(err)) return
    yield {
      kind: 'error',
      data: { message: 'fetch_failed', detail: String(err) },
    }
    yield { kind: 'done', data: { reason: 'fetch_failed' } }
    return
  }

  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => '')
    yield {
      kind: 'error',
      data: {
        message: `http_${response.status}`,
        detail: text || response.statusText,
      },
    }
    yield { kind: 'done', data: { reason: `http_${response.status}` } }
    return
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  try {
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx: number
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        const evt = parseSseBlock(block)
        if (evt) {
          yield evt
          if (evt.kind === 'done') {
            try { await reader.cancel() } catch { /* noop */ }
            return
          }
        }
      }
    }
    if (buffer.trim().length > 0) {
      const evt = parseSseBlock(buffer)
      if (evt) yield evt
    }
  } catch (err) {
    if (isAbort(err)) return
    yield {
      kind: 'error',
      data: { message: 'stream_error', detail: String(err) },
    }
    yield { kind: 'done', data: { reason: 'stream_error' } }
  }
}

function parseSseBlock(block: string): MiraEvent | null {
  let eventName = 'message'
  const dataLines: string[] = []
  for (const line of block.split('\n')) {
    if (line.startsWith('event:')) eventName = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim())
  }
  if (dataLines.length === 0) return null
  const raw = dataLines.join('\n')
  switch (eventName) {
    case 'thought': {
      let text = raw
      try {
        const obj = JSON.parse(raw)
        if (typeof obj === 'string') text = obj
      } catch { /* keep raw */ }
      return { kind: 'thought', data: text }
    }
    case 'tool_intent':
      try {
        return { kind: 'tool_intent', data: JSON.parse(raw) as ToolIntent }
      } catch (err) {
        return { kind: 'error', data: { message: 'parse_tool_intent', detail: String(err) } }
      }
    case 'tool_result':
      try {
        return { kind: 'tool_result', data: JSON.parse(raw) as ToolResult }
      } catch (err) {
        return { kind: 'error', data: { message: 'parse_tool_result', detail: String(err) } }
      }
    case 'tool_error':
      try {
        return { kind: 'tool_error', data: JSON.parse(raw) as ToolError }
      } catch {
        return { kind: 'tool_error', data: { tool: 'unknown', code: 'parse', message: raw } }
      }
    case 'error':
      try {
        return { kind: 'error', data: JSON.parse(raw) }
      } catch {
        return { kind: 'error', data: { message: raw } }
      }
    case 'done':
      try {
        return { kind: 'done', data: JSON.parse(raw) }
      } catch {
        return { kind: 'done', data: {} }
      }
    default:
      return null
  }
}

// ── Voice (TTS + STT) ──────────────────────────────────

async function parseUnavailable(response: Response): Promise<ServiceUnavailable> {
  try {
    const body = (await response.json()) as {
      detail?: ServiceUnavailable | string
    }
    if (body.detail && typeof body.detail === 'object') return body.detail
    return {
      detail: typeof body.detail === 'string' ? body.detail : 'unavailable',
      install_hint: '',
    }
  } catch {
    return { detail: 'unavailable', install_hint: '' }
  }
}

export interface TtsResult {
  blob: Blob
  url: string
}

export async function ttsSynthesise(
  text: string,
  speed: number,
  opts: ApiOptions = {},
): Promise<TtsResult> {
  const response = await fetch(`${BASE}/voice/tts`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, speed }),
    signal: opts.signal,
  })
  if (response.status === 503) {
    const info = await parseUnavailable(response)
    throw new ServiceUnavailableError(info.detail, info)
  }
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new ApiError(
      `${response.status} ${response.statusText}${text ? ` \u2014 ${text}` : ''}`,
      response.status,
    )
  }
  const blob = await response.blob()
  return { blob, url: URL.createObjectURL(blob) }
}

export async function sttTranscribe(
  blob: Blob,
  opts: ApiOptions = {},
): Promise<SttResponse> {
  const form = new FormData()
  const filename =
    blob.type.includes('webm') ? 'recording.webm'
    : blob.type.includes('ogg') ? 'recording.ogg'
    : 'recording.wav'
  form.append('audio', blob, filename)
  const response = await fetch(`${BASE}/voice/stt`, {
    method: 'POST',
    body: form,
    signal: opts.signal,
  })
  if (response.status === 503) {
    const info = await parseUnavailable(response)
    throw new ServiceUnavailableError(info.detail, info)
  }
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new ApiError(
      `${response.status} ${response.statusText}${text ? ` \u2014 ${text}` : ''}`,
      response.status,
    )
  }
  return (await response.json()) as SttResponse
}

// ── Admin (runtime LLM options) ────────────────────────

export interface LlmOptions {
  model: string
  think: boolean
  temperature: number
}

export interface OllamaStatus {
  reachable: boolean
  base_url: string
  models: string[]
  configured_available: boolean
  error: string | null
}

export interface OllamaStatusResponse {
  options: LlmOptions
  ollama: OllamaStatus
}

export const getOllamaStatus = (
  opts?: ApiOptions,
): Promise<OllamaStatusResponse> =>
  request<OllamaStatusResponse>('/admin/ollama', opts)

export const updateLlmOptions = (
  patch: Partial<LlmOptions>,
): Promise<{ options: LlmOptions }> =>
  request<{ options: LlmOptions }>('/admin/ollama', {
    method: 'PUT',
    body: JSON.stringify(patch),
  })

export { ApiError, isAbort }
