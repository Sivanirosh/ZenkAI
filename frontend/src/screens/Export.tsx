// Export — the CSV generation surface for the external SRS (decision 0001).
//
// Lists every queued word, with inline-editable question/answer/extra_tags
// overrides, a per-row annotate button for words missing a definition, a
// bulk "Alle annotieren" button that fills gaps across the whole queue, and
// a header-mounted export button that writes the CSV, starts a browser
// download, and flips queued rows to exported.

'use client'

import { useCallback, useEffect, useMemo, useState } from 'react'
import * as api from '@/lib/api'
import type { VocabEntry } from '@/lib/types'
import { useSession } from '@/store/session'
import { useVocab } from '@/hooks/useVocab'

function autoQuestion(entry: VocabEntry): string {
  if (entry.question_override && entry.question_override.trim()) {
    return entry.question_override.trim()
  }
  if (entry.pos === 'NOUN') {
    const article =
      entry.gender === 'm'
        ? 'der'
        : entry.gender === 'f'
          ? 'die'
          : entry.gender === 'n'
            ? 'das'
            : null
    if (article) return `${article} ${entry.lemma} (${entry.gender})`
    return `${entry.lemma} (n)`
  }
  if (entry.pos === 'VERB') return `${entry.lemma} (v)`
  if (entry.pos === 'ADJ') return `${entry.lemma} (adj)`
  return entry.lemma
}

function autoAnswer(entry: VocabEntry): string {
  const override = entry.answer_override?.trim()
  if (override) return override
  const de = entry.definition_de?.trim()
  const en = entry.definition_en?.trim()
  return [de, en].filter(Boolean).join('\n')
}

type RowDraft = {
  question: string
  answer: string
  extra_tags: string
}

function draftFromEntry(entry: VocabEntry): RowDraft {
  return {
    question: entry.question_override ?? '',
    answer: entry.answer_override ?? '',
    extra_tags: entry.extra_tags ?? '',
  }
}

export function Export() {
  const setTab = useSession((s) => s.setTab)
  const bulkSetVocabStatus = useSession((s) => s.bulkSetVocabStatus)
  const { remove: removeFromQueue } = useVocab()

  const [entries, setEntries] = useState<VocabEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [drafts, setDrafts] = useState<Record<string, RowDraft>>({})
  const [annotating, setAnnotating] = useState<Record<string, boolean>>({})
  const [bulkAnnotating, setBulkAnnotating] = useState(false)
  const [bulkProgress, setBulkProgress] = useState<{
    done: number
    total: number
  } | null>(null)
  const [exporting, setExporting] = useState(false)
  const [exportMessage, setExportMessage] = useState<string | null>(null)

  const load = useCallback(async () => {
    setLoading(true)
    setLoadError(null)
    try {
      const list = await api.vocabList('queued')
      setEntries(list)
      const initial: Record<string, RowDraft> = {}
      for (const e of list) initial[e.word_id] = draftFromEntry(e)
      setDrafts(initial)
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  const needsAnnotation = useMemo(
    () => entries.filter((e) => !autoAnswer(e)).length,
    [entries],
  )

  // Rows that are missing at least one of {DE, EN} and have no override.
  // Used by the bulk-annotate action so users can fill the whole queue in
  // one go instead of clicking every row individually.
  const annotatableEntries = useMemo(
    () =>
      entries.filter((e) => {
        if (e.answer_override?.trim()) return false
        const hasDe = Boolean(e.definition_de?.trim())
        const hasEn = Boolean(e.definition_en?.trim())
        return !hasDe || !hasEn
      }),
    [entries],
  )

  const updateDraft = (wordId: string, patch: Partial<RowDraft>) => {
    setDrafts((prev) => ({
      ...prev,
      [wordId]: { ...prev[wordId], ...patch },
    }))
  }

  const persistOverrides = async (entry: VocabEntry) => {
    const draft = drafts[entry.word_id]
    if (!draft) return
    const body = {
      question: draft.question.trim() === '' ? null : draft.question.trim(),
      answer: draft.answer.trim() === '' ? null : draft.answer.trim(),
      extra_tags:
        draft.extra_tags.trim() === '' ? null : draft.extra_tags.trim(),
    }
    // Skip if nothing changed.
    if (
      (body.question ?? '') === (entry.question_override ?? '') &&
      (body.answer ?? '') === (entry.answer_override ?? '') &&
      (body.extra_tags ?? '') === (entry.extra_tags ?? '')
    ) {
      return
    }
    try {
      const updated = await api.vocabUpdateOverrides(entry.word_id, body)
      setEntries((prev) =>
        prev.map((e) => (e.word_id === updated.word_id ? updated : e)),
      )
    } catch (err) {
      setExportMessage(
        `Speichern fehlgeschlagen: ${
          err instanceof Error ? err.message : String(err)
        }`,
      )
    }
  }

  const handleAnnotate = async (entry: VocabEntry) => {
    setAnnotating((p) => ({ ...p, [entry.word_id]: true }))
    try {
      const updated = await api.vocabAnnotate(entry.word_id)
      setEntries((prev) =>
        prev.map((e) => (e.word_id === updated.word_id ? updated : e)),
      )
      setDrafts((prev) => ({
        ...prev,
        [updated.word_id]: draftFromEntry(updated),
      }))
    } catch (err) {
      setExportMessage(
        `Annotation fehlgeschlagen: ${
          err instanceof Error ? err.message : String(err)
        }`,
      )
    } finally {
      setAnnotating((p) => {
        const next = { ...p }
        delete next[entry.word_id]
        return next
      })
    }
  }

  // Sequentially annotate every row missing a definition. Sequential keeps
  // Ollama responsive and makes progress legible; the per-row state also
  // updates as each call completes so partial results are visible.
  const handleBulkAnnotate = async () => {
    const targets = annotatableEntries
    if (targets.length === 0 || bulkAnnotating) return
    setBulkAnnotating(true)
    setBulkProgress({ done: 0, total: targets.length })
    setExportMessage(null)
    let failed = 0
    for (let i = 0; i < targets.length; i += 1) {
      const entry = targets[i]
      setAnnotating((p) => ({ ...p, [entry.word_id]: true }))
      try {
        const updated = await api.vocabAnnotate(entry.word_id)
        setEntries((prev) =>
          prev.map((e) => (e.word_id === updated.word_id ? updated : e)),
        )
        setDrafts((prev) => ({
          ...prev,
          [updated.word_id]: draftFromEntry(updated),
        }))
      } catch {
        failed += 1
      } finally {
        setAnnotating((p) => {
          const next = { ...p }
          delete next[entry.word_id]
          return next
        })
        setBulkProgress({ done: i + 1, total: targets.length })
      }
    }
    setBulkAnnotating(false)
    setBulkProgress(null)
    setExportMessage(
      failed === 0
        ? `${targets.length} W\u00f6rter annotiert.`
        : `${targets.length - failed} von ${targets.length} annotiert (${failed} fehlgeschlagen).`,
    )
  }

  const handleRemove = async (entry: VocabEntry) => {
    try {
      await removeFromQueue(entry.word_id)
      setEntries((prev) => prev.filter((e) => e.word_id !== entry.word_id))
    } catch (err) {
      setExportMessage(
        `Entfernen fehlgeschlagen: ${
          err instanceof Error ? err.message : String(err)
        }`,
      )
    }
  }

  const readyToExport = entries.length > 0 && needsAnnotation === 0

  const handleExport = async () => {
    if (!readyToExport) return
    setExporting(true)
    setExportMessage(null)
    try {
      const exportedIds = entries.map((e) => e.word_id)
      const ok = await api.vocabExport()
      if (!ok) {
        setExportMessage('Keine Karten im Export.')
      } else {
        const map: Record<string, 'exported'> = {}
        for (const id of exportedIds) map[id] = 'exported'
        bulkSetVocabStatus(map)
        setEntries([])
        setDrafts({})
        setExportMessage(`${exportedIds.length} Karten exportiert.`)
      }
    } catch (err) {
      setExportMessage(
        `Export fehlgeschlagen: ${
          err instanceof Error ? err.message : String(err)
        }`,
      )
    } finally {
      setExporting(false)
    }
  }

  return (
    <div>
      <button className="back-btn" type="button" onClick={() => setTab('rd')}>
        {'\u2190'} Zur{'\u00FC'}ck zum Leseraum
      </button>

      <div
        className="card"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: 10,
        }}
      >
        <div>
          <div className="card-label">Export-Warteschlange</div>
          <div style={{ fontSize: 18, fontWeight: 600 }}>
            {entries.length} W{'\u00F6'}rter zum Export
            {needsAnnotation > 0 ? (
              <span
                style={{
                  marginLeft: 10,
                  fontSize: 13,
                  color: 'var(--amber-800)',
                  fontWeight: 400,
                }}
              >
                {'\u00b7'} {needsAnnotation} brauchen Annotation
              </span>
            ) : null}
            {annotatableEntries.length > 0 && needsAnnotation === 0 ? (
              <span
                style={{
                  marginLeft: 10,
                  fontSize: 13,
                  color: 'var(--color-text-secondary)',
                  fontWeight: 400,
                }}
              >
                {'\u00b7'} {annotatableEntries.length} nur teilweise annotiert
              </span>
            ) : null}
          </div>
        </div>
        <div
          style={{
            display: 'flex',
            gap: 8,
            flexWrap: 'wrap',
            justifyContent: 'flex-end',
          }}
        >
          <button
            className="btn-secondary"
            type="button"
            onClick={() => void load()}
            disabled={bulkAnnotating || exporting}
          >
            Aktualisieren
          </button>
          <button
            className="btn-secondary"
            type="button"
            onClick={() => void handleBulkAnnotate()}
            disabled={
              annotatableEntries.length === 0 || bulkAnnotating || exporting
            }
            title={
              annotatableEntries.length === 0
                ? 'Alle W\u00f6rter sind vollst\u00e4ndig annotiert'
                : `${annotatableEntries.length} W\u00f6rter annotieren`
            }
          >
            {bulkAnnotating && bulkProgress
              ? `Annotiere ${bulkProgress.done}/${bulkProgress.total}\u2026`
              : `Alle annotieren${
                  annotatableEntries.length > 0
                    ? ` (${annotatableEntries.length})`
                    : ''
                }`}
          </button>
          <button
            className="btn-primary"
            type="button"
            onClick={() => void handleExport()}
            disabled={!readyToExport || exporting || bulkAnnotating}
            title={
              !readyToExport && entries.length > 0
                ? `${needsAnnotation} W\u00f6rter brauchen noch eine Annotation`
                : undefined
            }
          >
            {exporting
              ? 'Exportiere\u2026'
              : entries.length > 0
                ? `Export ${entries.length} Karten als CSV`
                : 'Export CSV'}
          </button>
        </div>
      </div>

      {loading ? (
        <div className="skeleton">L{'\u00E4'}dt Warteschlange{'\u2026'}</div>
      ) : loadError ? (
        <div className="empty-state">Fehler: {loadError}</div>
      ) : entries.length === 0 ? (
        <div className="empty-state">
          Noch keine W{'\u00F6'}rter in der Warteschlange. Klicke W{'\u00F6'}rter
          im Leseraum, um sie hier zu sammeln.
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {entries.map((entry) => {
            const draft = drafts[entry.word_id] ?? draftFromEntry(entry)
            const ans = autoAnswer(entry)
            const needsAnn = !ans
            const hasOverride = Boolean(entry.answer_override?.trim())
            const hasDe = Boolean(entry.definition_de?.trim())
            const hasEn = Boolean(entry.definition_en?.trim())
            const partialAnnotation = !hasOverride && (hasDe !== hasEn)
            return (
              <div
                key={entry.word_id}
                className="card"
                style={{ display: 'flex', flexDirection: 'column', gap: 8 }}
              >
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'baseline',
                    gap: 8,
                    flexWrap: 'wrap',
                  }}
                >
                  <div>
                    <strong style={{ fontSize: 16 }}>
                      {autoQuestion(entry)}
                    </strong>
                    <span
                      style={{
                        marginLeft: 8,
                        fontSize: 12,
                        color: 'var(--color-text-secondary)',
                      }}
                    >
                      {entry.pos ?? '\u2014'}
                    </span>
                  </div>
                  <div style={{ display: 'flex', gap: 6 }}>
                    {needsAnn ? (
                      <button
                        className="btn-primary"
                        type="button"
                        onClick={() => void handleAnnotate(entry)}
                        disabled={Boolean(annotating[entry.word_id])}
                      >
                        {annotating[entry.word_id]
                          ? 'Annotiert\u2026'
                          : 'Annotieren'}
                      </button>
                    ) : null}
                    <button
                      className="btn-secondary"
                      type="button"
                      onClick={() => void handleRemove(entry)}
                      title="Aus Warteschlange entfernen"
                    >
                      {'\u00D7'}
                    </button>
                  </div>
                </div>

                {entry.source_sentence ? (
                  <div
                    style={{
                      fontSize: 13,
                      fontStyle: 'italic',
                      color: 'var(--color-text-secondary)',
                      lineHeight: 1.55,
                    }}
                  >
                    {'\u201E'}
                    {entry.source_sentence.length > 240
                      ? entry.source_sentence.slice(0, 237) + '\u2026'
                      : entry.source_sentence}
                    {'\u201C'}
                  </div>
                ) : null}

                {ans ? (
                  <div
                    style={{
                      fontSize: 13,
                      lineHeight: 1.55,
                      padding: '8px 10px',
                      background: 'var(--color-bg-muted, rgba(0,0,0,0.03))',
                      borderRadius: 6,
                      whiteSpace: 'pre-line',
                    }}
                  >
                    {ans}
                    {partialAnnotation ? (
                      <div
                        style={{
                          marginTop: 6,
                          fontSize: 11,
                          color: 'var(--amber-800)',
                        }}
                      >
                        {'\u00b7'} Nur {hasDe ? 'Deutsch' : 'Englisch'}{' '}
                        vorhanden {'\u2014'} Annotieren f{'\u00FC'}r beide
                        Sprachen.
                      </div>
                    ) : null}
                  </div>
                ) : null}
                {partialAnnotation ? (
                  <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                      className="btn-secondary"
                      type="button"
                      onClick={() => void handleAnnotate(entry)}
                      disabled={Boolean(annotating[entry.word_id])}
                    >
                      {annotating[entry.word_id]
                        ? 'Annotiert\u2026'
                        : 'Beide Sprachen holen'}
                    </button>
                  </div>
                ) : null}

                <div
                  style={{ display: 'flex', gap: 8, flexDirection: 'column' }}
                >
                  <label style={{ fontSize: 11, fontWeight: 600 }}>
                    Frage (override)
                    <input
                      type="text"
                      className="text-input"
                      value={draft.question}
                      placeholder={autoQuestion(entry)}
                      onChange={(e) =>
                        updateDraft(entry.word_id, { question: e.target.value })
                      }
                      onBlur={() => void persistOverrides(entry)}
                    />
                  </label>
                  <label style={{ fontSize: 11, fontWeight: 600 }}>
                    Antwort (override — lass leer f{'\u00FC'}r Auto-DE+EN)
                    <textarea
                      className="text-input"
                      value={draft.answer}
                      placeholder={
                        ans || 'Noch keine Definition \u2014 Annotieren'
                      }
                      rows={3}
                      onChange={(e) =>
                        updateDraft(entry.word_id, { answer: e.target.value })
                      }
                      onBlur={() => void persistOverrides(entry)}
                    />
                  </label>
                  <label style={{ fontSize: 11, fontWeight: 600 }}>
                    Extra-Tags (comma-separated)
                    <input
                      type="text"
                      className="text-input"
                      value={draft.extra_tags}
                      placeholder="z.B. kafka,expressionism"
                      onChange={(e) =>
                        updateDraft(entry.word_id, {
                          extra_tags: e.target.value,
                        })
                      }
                      onBlur={() => void persistOverrides(entry)}
                    />
                  </label>
                </div>
              </div>
            )
          })}
        </div>
      )}

      {exportMessage ? (
        <div
          className="card"
          style={{
            marginTop: 12,
            fontSize: 13,
            color: 'var(--color-text-secondary)',
          }}
        >
          {exportMessage}
        </div>
      ) : null}
    </div>
  )
}
