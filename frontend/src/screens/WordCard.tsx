// Wortkarte: deep word detail — definition, grammar, literary context.
//
// Reads annotations and word records from the session cache so tab switches
// are instant; revalidates in the background when a new word is selected.

'use client'

import { useCallback, useMemo } from 'react'
import * as api from '@/lib/api'
import type { Word, WordAnnotation } from '@/lib/types'
import { GrammarPill } from '@/components/GrammarPill'
import { useSession } from '@/store/session'
import { useVocab } from '@/hooks/useVocab'
import { useCachedResource } from '@/hooks/useCachedResource'

interface WordPayload {
  annotation: WordAnnotation
  record: Word | null
}

export function WordCard() {
  const selected = useSession((s) => s.selectedWord)
  const setTab = useSession((s) => s.setTab)
  const clear = useSession((s) => s.clearSelectedWord)
  const openWord = useSession((s) => s.openWord)
  const wordCache = useSession((s) => s.wordCache)
  const cacheWord = useSession((s) => s.cacheWord)
  const annotationCache = useSession((s) => s.annotationCache)
  const cacheAnnotation = useSession((s) => s.cacheAnnotation)
  const { getStatus, enqueue, markKnown, remove } = useVocab()

  // Annotation cache key: word + context. The sentence affects the literary
  // note so it has to be part of the key; we clamp it to keep keys bounded.
  const annotationKey = useMemo(() => {
    if (!selected) return null
    return `${selected.wordId ?? selected.surfaceForm}|${selected.sentence.slice(0, 120)}`
  }, [selected])

  const cachedAnnotation = annotationKey
    ? annotationCache[annotationKey] ?? null
    : null
  const cachedWord =
    selected?.wordId ? wordCache[selected.wordId] ?? null : null

  const fetchWordPayload = useCallback(
    async (signal: AbortSignal): Promise<WordPayload> => {
      const snapshot = selected!
      const [annotation, record] = await Promise.all([
        api.getWordAnnotation(
          {
            word: snapshot.surfaceForm,
            sentence: snapshot.sentence,
            grammatical_role: snapshot.grammaticalRole,
            case_label: snapshot.caseLabel,
          },
          { signal },
        ),
        snapshot.wordId
          ? api.getWord(snapshot.wordId, { signal }).catch(() => null)
          : Promise.resolve(null),
      ])
      return { annotation, record }
    },
    [selected],
  )

  const onData = useCallback(
    ({ annotation, record }: WordPayload) => {
      if (annotationKey) cacheAnnotation(annotationKey, annotation)
      if (record) cacheWord(record)
    },
    [annotationKey, cacheAnnotation, cacheWord],
  )

  // Pack the caches into a single pseudo-cache for the hook; presence of
  // both means "no loading skeleton needed".
  const cached: WordPayload | null =
    cachedAnnotation && (!selected?.wordId || cachedWord)
      ? { annotation: cachedAnnotation, record: cachedWord }
      : null

  const { loading, error } = useCachedResource<WordPayload>({
    key: annotationKey,
    cached,
    fetcher: fetchWordPayload,
    onData,
  })

  const annotation = cachedAnnotation
  const wordRecord = cachedWord

  if (!selected) {
    return (
      <div className="empty-state">
        W{'\u00E4'}hle ein Wort im Leseraum um die Wortkarte zu {'\u00F6'}ffnen.
      </div>
    )
  }

  const vocabStatus = selected.wordId ? getStatus(selected.wordId) : 'new'

  const genderLabel = wordRecord?.gender
    ? wordRecord.gender === 'n'
      ? 'Neutrum'
      : wordRecord.gender === 'm'
        ? 'Maskulinum'
        : 'Femininum'
    : null

  const stateLabel =
    vocabStatus === 'queued'
      ? 'In der Warteschlange'
      : vocabStatus === 'known'
        ? 'Bekannt'
        : vocabStatus === 'exported'
          ? 'Exportiert'
          : 'Neu \u00b7 nicht markiert'

  return (
    <div>
      <button
        className="back-btn"
        type="button"
        onClick={() => {
          clear()
          setTab('rd')
        }}
      >
        {'\u2190'} Zur{'\u00FC'}ck zum Text
      </button>

      <div className="card">
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: 14,
          }}
        >
          <div>
            <div className="word-hero">{selected.surfaceForm}</div>
            <div className="word-grammar-info">
              {wordRecord?.lemma ?? selected.surfaceForm}
              {genderLabel ? ` \u00b7 ${genderLabel}` : ''}
              {wordRecord?.pos ? ` \u00b7 ${wordRecord.pos}` : ''}
            </div>
          </div>
          <span
            className="badge badge-amber"
            style={{ whiteSpace: 'nowrap', flexShrink: 0, marginLeft: 10 }}
          >
            {stateLabel}
          </span>
        </div>

      </div>

      <div className="card">
        <div className="card-label">Bedeutung</div>
        {loading && !annotation ? (
          <div className="skeleton">L{'\u00E4'}dt Erkl{'\u00E4'}rung{'\u2026'}</div>
        ) : error && !annotation ? (
          <div className="empty-state">Fehler: {error}</div>
        ) : annotation ? (
          <>
            <div style={{ fontSize: 15, marginBottom: 6, lineHeight: 1.55 }}>
              {annotation.definition_de}
            </div>
            <div
              style={{
                fontSize: 13,
                color: 'var(--color-text-secondary)',
                marginBottom: 6,
              }}
            >
              {annotation.definition_en}
            </div>
            {annotation.etymology ? (
              <div
                style={{
                  fontSize: 12,
                  color: 'var(--color-text-secondary)',
                  fontStyle: 'italic',
                }}
              >
                {annotation.etymology}
              </div>
            ) : null}
            {annotation.source === 'mock' ? (
              <div style={{ marginTop: 8, fontSize: 10, color: 'var(--color-text-tertiary)' }}>
                Offline-Erkl{'\u00E4'}rung {'\u2014'} Ollama starten f{'\u00FC'}r AI.
              </div>
            ) : null}
          </>
        ) : null}
      </div>

      <div className="card">
        <div className="card-label">Grammatik in dieser Passage</div>
        <div className="grammar-grid">
          <GrammarPill
            label="Kasus"
            value={selected.caseLabel ?? '\u2014'}
          />
          <GrammarPill
            label="Rolle"
            value={selected.grammaticalRole ?? '\u2014'}
          />
          <GrammarPill label="Genus" value={genderLabel ?? '\u2014'} />
        </div>
      </div>

      {annotation?.literary_note ? (
        <div className="card">
          <div className="card-label">Literarischer Kontext</div>
          <div className="lit-quote">
            {'\u201E'}
            {selected.sentence}
            {'\u201C'}
          </div>
          <div
            style={{
              fontSize: 13,
              color: 'var(--color-text-secondary)',
              lineHeight: 1.65,
            }}
          >
            {annotation.literary_note}
          </div>
        </div>
      ) : null}

      {annotation && annotation.related_words.length > 0 ? (
        <div className="card">
          <div className="card-label">Verwandte W{'\u00F6'}rter</div>
          <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
            {annotation.related_words.map((w) => (
              <button
                className="chip"
                key={w}
                type="button"
                onClick={() =>
                  openWord({
                    wordId: null,
                    surfaceForm: w,
                    sentence: selected.sentence,
                    grammaticalRole: null,
                    caseLabel: null,
                  })
                }
              >
                {w}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      <div className="btn-row" style={{ flexWrap: 'wrap' }}>
        <button
          className="btn-secondary"
          type="button"
          onClick={() => {
            clear()
            setTab('rd')
          }}
        >
          Schlie{'\u00DF'}en
        </button>
        {selected.wordId ? (
          <>
            {vocabStatus === 'queued' || vocabStatus === 'exported' ? (
              <button
                className="btn-secondary"
                type="button"
                onClick={async () => {
                  if (selected.wordId) await remove(selected.wordId)
                }}
              >
                Aus Warteschlange entfernen
              </button>
            ) : null}
            {vocabStatus !== 'known' ? (
              <button
                className="btn-secondary"
                type="button"
                onClick={async () => {
                  if (selected.wordId) await markKnown(selected.wordId)
                }}
              >
                Als bekannt markieren
              </button>
            ) : null}
            {vocabStatus !== 'queued' ? (
              <button
                className="btn-primary"
                type="button"
                onClick={async () => {
                  if (selected.wordId) {
                    await enqueue(
                      selected.wordId,
                      null,
                      selected.sentence,
                    )
                  }
                  clear()
                  setTab('rd')
                }}
              >
                In Warteschlange
              </button>
            ) : null}
          </>
        ) : null}
      </div>
    </div>
  )
}
