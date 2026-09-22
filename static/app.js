// ZenkAI — single-file frontend. No framework, no build step.
//
// Tabs: Bibliothek (lib) · Leseraum (rd) · Wortkarte (wc) · Export (ex) · Stimme (vx)
//
// Reader word gestures (decision 0001):
//   click → enqueue · shift/long-press → mark known · alt/meta → Wortkarte
'use strict'

const BASE = '/api/v1'

// ── tiny DOM + fetch helpers ────────────────────────────────────────────

const $ = (sel, root = document) => root.querySelector(sel)
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)]

const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]))

class ApiError extends Error {
  constructor(message, status) {
    super(message); this.status = status
  }
}

async function api(path, init = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
    ...init,
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new ApiError(`${res.status} ${res.statusText}${text ? ` — ${text}` : ''}`, res.status)
  }
  return res.status === 204 ? undefined : res.json()
}

const debounce = (fn, ms) => {
  let t
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms) }
}

// ── persisted state (navigation + prefs only) ───────────────────────────

const DEFAULT_PREFS = { fontScale: 1, focusMode: false, ttsSpeed: 1.0, ttsAutoPlay: true }

const S = {
  tab: 'lib',
  currentWorkId: null,
  currentParagraphId: null,
  epochFilter: null,
  selectedWord: null,           // {wordId, surfaceForm, sentence, grammaticalRole, caseLabel}
  vocabStatus: {},              // wordId → 'queued' | 'known' | 'exported'; absence = new
  readerPrefs: { ...DEFAULT_PREFS },
  lastParagraphByWork: {},
}

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem('zenkai-state') || '{}')
    Object.assign(S, saved, { readerPrefs: { ...DEFAULT_PREFS, ...saved.readerPrefs } })
  } catch { /* corrupt state — start fresh */ }
}

const saveState = debounce(() => {
  const { tab, currentWorkId, currentParagraphId, epochFilter, vocabStatus,
    readerPrefs, lastParagraphByWork } = S
  localStorage.setItem('zenkai-state', JSON.stringify(
    { tab, currentWorkId, currentParagraphId, epochFilter, vocabStatus, readerPrefs, lastParagraphByWork }))
}, 250)

// ── in-memory caches (never persisted; ingestion changes must not go stale)

const mem = {
  works: null,
  counts: { queued: 0, known: 0, exported: 0 },
  chapters: new Map(),        // workId → Chapter[]
  chapterParas: new Map(),    // `${workId}:${chapter}` → Paragraph[]
  paraById: new Map(),        // id → ParagraphWithTokens
  wordById: new Map(),        // id → Word
  annotation: new Map(),      // `${wordId|surface}|${sentence.slice(0,120)}` → WordAnnotation
  chat: new Map(),            // paragraphId → [{role, content}]
  scroll: new Map(),          // `${workId}:${chapter}:${pageIdx}` → scrollTop
}

// ── vocab harvester ops (optimistic, mirrors the old useVocab) ──────────

const getStatus = (wordId) => S.vocabStatus[wordId] ?? 'new'

async function vocabEnqueue(wordId, paragraphId, sentence) {
  const prev = getStatus(wordId)
  if (prev !== 'exported') S.vocabStatus[wordId] = 'queued'
  try {
    const entry = await api('/vocab/queue', {
      method: 'POST',
      body: JSON.stringify({ word_id: wordId, paragraph_id: paragraphId, sentence }),
    })
    S.vocabStatus[wordId] = entry.status
  } catch (err) {
    if (prev === 'new') delete S.vocabStatus[wordId]; else S.vocabStatus[wordId] = prev
    throw err
  } finally { afterVocabMutation() }
}

async function vocabMarkKnown(wordId) {
  const prev = getStatus(wordId)
  S.vocabStatus[wordId] = 'known'
  try {
    const entry = await api('/vocab/known', { method: 'POST', body: JSON.stringify({ word_id: wordId }) })
    S.vocabStatus[wordId] = entry.status
  } catch (err) {
    S.vocabStatus[wordId] = prev
    throw err
  } finally { afterVocabMutation() }
}

async function vocabRemove(wordId) {
  const prev = getStatus(wordId)
  delete S.vocabStatus[wordId]
  try {
    await api(`/vocab/queue/${encodeURIComponent(wordId)}`, { method: 'DELETE' })
  } catch (err) {
    S.vocabStatus[wordId] = prev
    throw err
  } finally { afterVocabMutation() }
}

function afterVocabMutation() {
  saveState()
  refreshCounts()
  // Re-paint word chips in the open reader / card so triage colors stay live.
  if (S.tab === 'rd') paintWordStatuses()
  if (S.tab === 'wc') renderWordCard()
}

async function hydrateStatuses(wordIds) {
  const missing = [...new Set(wordIds)].filter((id) => !(id in S.vocabStatus))
  if (missing.length === 0) return
  try {
    const { statuses } = await api('/vocab/statuses', { method: 'POST', body: JSON.stringify({ word_ids: missing }) })
    Object.assign(S.vocabStatus, statuses)
    saveState(); paintWordStatuses()
  } catch { /* best-effort hydration */ }
}

async function refreshCounts() {
  try {
    mem.counts = await api('/vocab/counts')
    const pill = $('#queue-pill')
    if (pill) pill.textContent = `${mem.counts.queued} ${mem.counts.queued === 1 ? 'Wort' : 'Wörter'} in Warteschlange`
    const exportBtn = $('#reader-export-btn')
    if (exportBtn) exportBtn.disabled = mem.counts.queued === 0
    const legend = $('#reader-counts')
    if (legend) legend.innerHTML = readerCountsHtml()
  } catch { /* best-effort */ }
}

// ── tab shell ────────────────────────────────────────────────────────────

const TABS = [
  { id: 'lib', label: 'Bibliothek' },
  { id: 'rd', label: 'Leseraum' },
  { id: 'wc', label: 'Wortkarte' },
  { id: 'ex', label: 'Export' },
  { id: 'vx', label: 'Stimme' },
]

function switchTab(tab) {
  S.tab = tab
  saveState()
  renderTabs()
  for (const t of TABS) $(`#screen-${t.id}`).style.display = t.id === tab ? '' : 'none'
  if (tab === 'lib') renderLibrary()
  if (tab === 'rd') initReader()
  if (tab === 'wc') renderWordCard()
  if (tab === 'ex') initExport()
  if (tab === 'vx') initVoice()
}

function renderTabs() {
  $('#tab-bar').innerHTML = TABS.map((t) =>
    `<button class="tab${S.tab === t.id ? ' active' : ''}" data-tab="${t.id}">${t.label}</button>`).join('')
}

// ═════════════════════════════════════════════════════════════ Bibliothek

const EPOCHS = ['Romantik', 'Expressionismus', 'Moderne', 'Volksliteratur']

async function renderLibrary() {
  const root = $('#screen-lib')
  root.innerHTML = `<div class="skeleton">Lädt Bibliothek…</div>`
  let works
  try {
    ;[works, mem.counts] = await Promise.all([api('/corpus/works'), api('/vocab/counts')])
    mem.works = works
  } catch (err) {
    root.innerHTML = `
      <div class="empty-state">
        <div style="margin-bottom:10px">Backend nicht erreichbar.</div>
        <div style="font-size:12px;color:var(--color-text-secondary);margin-bottom:12px">${esc(err.message)}</div>
        <div style="margin-bottom:12px">Starte das Backend mit <code>./scripts/run_backend.sh</code></div>
        <button class="btn-primary" id="lib-retry">Erneut versuchen</button>
      </div>`
    $('#lib-retry').onclick = () => renderLibrary()
    return
  }

  let search = ''
  const draw = () => {
    const filtered = works.filter((w) => {
      if (S.epochFilter && w.epoch !== S.epochFilter) return false
      if (search.trim()) {
        const n = search.toLowerCase()
        if (!w.title.toLowerCase().includes(n) && !w.author.toLowerCase().includes(n)) return false
      }
      return true
    })
    const activeBooks = works.filter((w) => w.progress.total_words > 0).length
    root.innerHTML = `
      <div class="search-row">
        <input type="text" id="lib-search" placeholder="Suchen..." style="flex:1;min-width:130px" value="${esc(search)}">
        <div class="filter-chips">
          <button class="chip${S.epochFilter === null ? ' selected' : ''}" data-epoch="">Alle</button>
          ${EPOCHS.map((e) => `<button class="chip${S.epochFilter === e ? ' selected' : ''}" data-epoch="${e}">${e}</button>`).join('')}
        </div>
      </div>
      <div class="metrics">
        <div class="metric-card"><div class="metric-label">Bekannte Wörter</div>
          <div class="metric-value">${mem.counts.known.toLocaleString('de-DE')}</div></div>
        <div class="metric-card"><div class="metric-label">Gelernte Wörter</div>
          <div class="metric-value">${(mem.counts.queued + mem.counts.exported).toLocaleString('de-DE')}</div></div>
        <div class="metric-card"><div class="metric-label">Aktive Bücher</div>
          <div class="metric-value">${activeBooks}</div></div>
      </div>
      ${filtered.length === 0
        ? `<div class="empty-state">Keine Bücher. Führe <code>python scripts/ingest_book.py --gutenberg-id 22367</code> aus.</div>`
        : `<div class="book-grid">${filtered.map(bookCardHtml).join('')}</div>`}`
    $('#lib-search').oninput = (e) => {
      search = e.target.value
      draw()
      // draw() rebuilds the DOM — put focus + caret back so typing flows.
      const input = $('#lib-search')
      input.focus()
      input.setSelectionRange(input.value.length, input.value.length)
    }
    $$('[data-epoch]', root).forEach((b) => b.onclick = () => { S.epochFilter = b.dataset.epoch || null; saveState(); draw() })
    $$('[data-open-work]', root).forEach((b) => b.onclick = () => openWork(b.dataset.openWork))
  }
  draw()
}

function bookCardHtml(w) {
  const p = w.progress
  const pct = p.total_words === 0 ? 0 : p.known_pct
  return `
    <div class="book-card" data-open-work="${esc(w.id)}" role="button" tabindex="0">
      <div class="book-spine" style="background:${esc(w.spine_color || '#b45309')}"></div>
      <div class="book-info">
        <div class="book-title">${esc(w.title)}</div>
        <div class="book-author">${esc(w.author)}${w.year ? ` · ${w.year}` : ''}</div>
        <div class="book-epoch">${esc(w.epoch ?? '')}</div>
        <div class="book-meta">${p.total_words === 0 ? 'Nicht begonnen' : `${pct.toFixed(0)}% bekannt`}</div>
        <div class="progress-bar"><div class="progress-fill" style="width:${pct}%"></div></div>
      </div>
    </div>`
}

async function openWork(workId) {
  try {
    const chapters = await api(`/corpus/works/${encodeURIComponent(workId)}/chapters`)
    mem.chapters.set(workId, chapters)
    const remembered = S.lastParagraphByWork[workId]
    const first = chapters[0]?.first_paragraph_id ?? null
    S.currentWorkId = workId
    S.currentParagraphId = remembered ?? first
    saveState()
    switchTab('rd')
  } catch (err) { alert(`Buch konnte nicht geöffnet werden: ${err.message}`) }
}

// ═════════════════════════════════════════════════════════════ Leseraum

// Client-side pagination: pack paragraphs into reader pages of ~350 words.
function buildPages(paragraphs, { targetWords = 350, minParas = 3, maxParas = 10 } = {}) {
  if (paragraphs.length === 0) return []
  const pages = []
  let ids = [], words = 0
  const flush = () => {
    if (ids.length === 0) return
    pages.push({ paragraphIds: ids, wordCount: words })
    ids = []; words = 0
  }
  for (const p of paragraphs) {
    ids.push(p.id)
    words += p.word_count ?? 0
    if (words >= targetWords || ids.length >= maxParas) flush()
  }
  flush()
  if (pages.length >= 2 && pages[pages.length - 1].paragraphIds.length < minParas) {
    const tail = pages.pop()
    const prev = pages[pages.length - 1]
    prev.paragraphIds.push(...tail.paragraphIds)
    prev.wordCount += tail.wordCount
  }
  return pages.map((p, i) => ({ ...p, index: i }))
}

const reader = {
  pages: [], chapter: 1, ready: false, loading: false, error: null,
  activeTtsId: null, initialised: false,
}

function readerScrollKey() {
  return `${S.currentWorkId}:${reader.chapter}:${pageIdx()}`
}

function pageIdx() {
  if (!S.currentParagraphId || reader.pages.length === 0) return 0
  const idx = reader.pages.findIndex((p) => p.paragraphIds.includes(S.currentParagraphId))
  return idx < 0 ? 0 : idx
}

async function initReader() {
  const root = $('#screen-rd')
  if (!S.currentWorkId) {
    root.innerHTML = `
      <div class="empty-state">Wähle ein Buch in der Bibliothek.
        <div style="margin-top:12px"><button class="btn-primary" id="rd-to-lib">Zur Bibliothek</button></div>
      </div>`
    $('#rd-to-lib').onclick = () => switchTab('lib')
    return
  }
  if (!reader.initialised) {
    reader.initialised = true
    window.addEventListener('keydown', readerKeys)
    window.addEventListener('scroll', () => {
      if (S.tab === 'rd' && reader.ready) mem.scroll.set(readerScrollKey(), window.scrollY)
    }, { passive: true })
  }
  await ensureChapters()
  try {
    await loadChapter()
  } catch (err) {
    $('#screen-rd').innerHTML = `
      <div class="empty-state"><div style="margin-bottom:10px">Fehler beim Laden: ${esc(err.message)}</div>
      <button class="btn-primary" id="rd-retry">Erneut versuchen</button></div>`
    $('#rd-retry').onclick = () => initReader()
    return
  }
  renderReader()
}

async function ensureChapters() {
  if (mem.chapters.has(S.currentWorkId)) return
  mem.chapters.set(S.currentWorkId, await api(`/corpus/works/${encodeURIComponent(S.currentWorkId)}/chapters`))
}

async function loadChapter() {
  // Resolve the current paragraph's chapter (fetch it if not cached yet).
  if (S.currentParagraphId && !mem.paraById.has(S.currentParagraphId)) {
    try {
      mem.paraById.set(S.currentParagraphId,
        await api(`/corpus/paragraphs/${encodeURIComponent(S.currentParagraphId)}`))
    } catch { /* unknown id — fall through to first chapter */ }
  }
  const para = S.currentParagraphId ? mem.paraById.get(S.currentParagraphId) : null
  const chapters = mem.chapters.get(S.currentWorkId) ?? []
  reader.chapter = para?.chapter ?? chapters[0]?.chapter ?? 1

  const key = `${S.currentWorkId}:${reader.chapter}`
  if (!mem.chapterParas.has(key)) {
    mem.chapterParas.set(key,
      await api(`/corpus/works/${encodeURIComponent(S.currentWorkId)}/chapters/${reader.chapter}/paragraphs`))
  }
  reader.pages = buildPages(mem.chapterParas.get(key))
}

async function fetchPageParagraphs(page) {
  const missing = page.paragraphIds.filter((id) => !mem.paraById.has(id))
  if (missing.length === 0) return
  const results = await api('/corpus/paragraphs/batch', { method: 'POST', body: JSON.stringify({ ids: missing }) })
  for (const p of results) mem.paraById.set(p.id, p)
  const wordIds = results.flatMap((p) => p.tokens.map((t) => t.word_id).filter(Boolean))
  hydrateStatuses(wordIds)
  // Prefetch the next page in the background so page turns feel instant.
  const next = reader.pages[page.index + 1]
  if (next) {
    const nextMissing = next.paragraphIds.filter((id) => !mem.paraById.has(id))
    if (nextMissing.length > 0) {
      api('/corpus/paragraphs/batch', { method: 'POST', body: JSON.stringify({ ids: nextMissing }) })
        .then((ps) => ps.forEach((p) => mem.paraById.set(p.id, p)))
        .catch(() => {})
    }
  }
}

function goToPage(idx) {
  if (reader.pages.length === 0) return
  const clamped = Math.min(Math.max(idx, 0), reader.pages.length - 1)
  S.currentParagraphId = reader.pages[clamped].paragraphIds[0]
  saveState()
  renderReader()
}

function goToChapter(ch) {
  const chapters = mem.chapters.get(S.currentWorkId) ?? []
  const target = chapters.find((c) => c.chapter === ch)
  if (!target) return
  S.currentParagraphId = target.first_paragraph_id
  saveState()
  loadChapter().then(renderReader).catch(() => renderReader())
}

function goPrevPage() {
  if (pageIdx() > 0) return goToPage(pageIdx() - 1)
  goToChapter(reader.chapter - 1)
}

function goNextPage() {
  if (pageIdx() < reader.pages.length - 1) return goToPage(pageIdx() + 1)
  goToChapter(reader.chapter + 1)
}

function readerKeys(e) {
  if (S.tab !== 'rd') return
  const t = e.target
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return
  if (e.key === 'Enter' && t?.closest?.('#reader-text [data-w]')) {
    t.click(); e.preventDefault()
  } else if (e.key === 'ArrowLeft') { e.shiftKey ? goToChapter(reader.chapter - 1) : goPrevPage(); e.preventDefault() }
  else if (e.key === 'ArrowRight') { e.shiftKey ? goToChapter(reader.chapter + 1) : goNextPage(); e.preventDefault() }
  else if (e.key === 'f' || e.key === 'F') {
    S.readerPrefs.focusMode = !S.readerPrefs.focusMode
    saveState(); renderReader()
  }
}

function readerCountsHtml() {
  const c = mem.counts
  return `
    <span class="badge" style="background:var(--amber-50);color:var(--amber-800)" title="In der Export-Warteschlange">${c.queued} markiert</span>
    <span class="badge badge-teal">${c.known} bekannt</span>
    ${c.exported > 0 ? `<span class="badge" style="background:var(--color-bg-secondary);color:var(--color-text-secondary)">${c.exported} exportiert</span>` : ''}`
}

async function renderReader() {
  const root = $('#screen-rd')
  const work = mem.works?.find((w) => w.id === S.currentWorkId)
  const page = reader.pages[pageIdx()] ?? null
  const chapters = mem.chapters.get(S.currentWorkId) ?? []
  const focus = S.readerPrefs.focusMode
  const chapterIdx = Math.max(chapters.findIndex((c) => c.chapter === reader.chapter), 0)
  const bookProgress = chapters.length && reader.pages.length
    ? Math.min((chapterIdx + pageIdx() / reader.pages.length) / chapters.length, 1) : 0

  document.body.classList.toggle('reader-focus-mode', focus)
  root.innerHTML = `
    <div class="reader${focus ? ' focus' : ''}" id="reader-root" style="--reader-font-scale:${S.readerPrefs.fontScale}">
      <div class="reader-progress" role="progressbar" aria-valuenow="${Math.round(bookProgress * 100)}">
        <div class="reader-progress-fill" style="width:${bookProgress * 100}%"></div>
      </div>
      ${focus ? '' : `
      <div class="reader-header">
        <div>
          <div class="reader-work">${esc(work?.author ?? '')}${work?.year ? ` · ${work.year}` : ''}</div>
          <div class="reader-title">${esc(work?.title ?? '')}</div>
        </div>
        <div style="display:flex;gap:8px;align-items:center">
          <span class="badge" id="queue-pill" style="background:var(--amber-50);color:var(--amber-800)"></span>
          <button class="btn-secondary" id="reader-export-btn">Export CSV</button>
        </div>
      </div>`}
      <div class="reader-nav-sticky-top">
        <div class="reader-nav">
          <div class="reader-nav-group">
            <button class="reader-nav-btn" id="nav-prev" title="Seite zurück (←)">←</button>
            <select class="reader-nav-select" id="nav-chapter" title="Kapitel">
              ${chapters.map((c) => `<option value="${c.chapter}" ${c.chapter === reader.chapter ? 'selected' : ''}>Kapitel ${c.chapter}</option>`).join('')}
            </select>
            <button class="reader-nav-btn" id="nav-next" title="Seite vor (→)">→</button>
          </div>
          <div class="reader-nav-group">
            <input class="reader-nav-page-input" id="nav-page" type="number" min="1"
                   max="${reader.pages.length}" value="${pageIdx() + 1}" title="Seite">
            <span class="reader-nav-muted">/ ${reader.pages.length}</span>
          </div>
          <div class="reader-nav-group">
            <div class="reader-nav-fonts">
              <button class="reader-nav-font" id="font-down" title="Kleiner">A−</button>
              <button class="reader-nav-font" id="font-up" title="Größer">A+</button>
            </div>
            <button class="reader-nav-focus ${focus ? 'active' : ''}" id="focus-toggle" title="Fokusmodus (F)">◉</button>
          </div>
        </div>
      </div>
      ${focus ? '' : `
      <div class="legend">
        <div class="legend-item"><div class="swatch" style="background:var(--amber-50);border:0.5px solid var(--amber-100)"></div>Neu — klicken = markieren</div>
        <div class="legend-item"><div style="width:14px;height:0;border-bottom:2.5px solid var(--amber-600);display:inline-block"></div>Markiert — im Export</div>
        <div class="legend-item"><div class="swatch" style="background:var(--color-bg-secondary)"></div>Bekannt — Shift-Klick</div>
      </div>`}
      <div class="literary-text reader-page" id="reader-text"></div>
      ${focus ? '' : `<div id="reader-counts" style="display:flex;gap:6px;margin-bottom:12px;flex-wrap:wrap">${readerCountsHtml()}</div>`}
      <div id="tts-bar-slot"></div>
    </div>`

  $('#nav-prev').onclick = goPrevPage
  $('#nav-next').onclick = goNextPage
  $('#nav-chapter').onchange = (e) => goToChapter(Number(e.target.value))
  $('#nav-page').onchange = (e) => goToPage(Number(e.target.value) - 1)
  $('#font-down').onclick = () => { S.readerPrefs.fontScale = Math.max(0.8, S.readerPrefs.fontScale - 0.1); saveState(); renderReader() }
  $('#font-up').onclick = () => { S.readerPrefs.fontScale = Math.min(1.6, S.readerPrefs.fontScale + 0.1); saveState(); renderReader() }
  $('#focus-toggle').onclick = () => { S.readerPrefs.focusMode = !S.readerPrefs.focusMode; saveState(); renderReader() }
  $('#reader-export-btn')?.addEventListener('click', () => switchTab('ex'))
  refreshCounts()

  // ── page body
  const textEl = $('#reader-text')
  if (!page) {
    textEl.innerHTML = `<div class="empty-state">Keine Seiten in diesem Kapitel.</div>`
    return
  }
  textEl.innerHTML = `<div class="skeleton">Lädt Seite…</div>`
  try {
    await fetchPageParagraphs(page)
  } catch (err) {
    textEl.innerHTML = `
      <div class="empty-state"><div style="margin-bottom:10px">Fehler beim Laden: ${esc(err.message)}</div>
      <button class="btn-primary" id="rd-retry">Erneut versuchen</button></div>`
    $('#rd-retry').onclick = () => renderReader()
    return
  }
  const paras = page.paragraphIds.map((id) => mem.paraById.get(id)).filter(Boolean)
  textEl.innerHTML = paras.map(paragraphHtml).join('')
  paintWordStatuses()
  reader.ready = true

  const saved = mem.scroll.get(readerScrollKey()) ?? 0
  window.scrollTo({ top: saved, behavior: 'auto' })

  tts.mountBar($('#tts-bar-slot'), paras.map((p) => ({ id: p.id, text: p.text })))
}

// Render one paragraph: walk tokens, slice the original text at char offsets
// so punctuation/whitespace between words stays typographically exact.
function paragraphHtml(p) {
  if (!p.tokens || p.tokens.length === 0) return `<p>${esc(p.text)}</p>`
  let html = '', cursor = 0
  for (const t of p.tokens) {
    if (t.char_start > cursor) html += esc(p.text.slice(cursor, t.char_start))
    html += t.word_id
      ? `<span class="w-new" data-w="${esc(t.word_id)}" tabindex="0" role="button">${esc(t.surface_form)}</span>`
      : esc(t.surface_form)
    cursor = t.char_end
  }
  if (cursor < p.text.length) html += esc(p.text.slice(cursor))
  return `<p id="p-${esc(p.id)}" data-pid="${esc(p.id)}">${html}</p>`
}

// Paint triage colors onto rendered word spans from S.vocabStatus.
function paintWordStatuses() {
  const cls = { new: 'w-new', queued: 'w-queued', exported: 'w-exported', known: 'w-known' }
  $$('#reader-text [data-w]').forEach((span) => {
    const status = getStatus(span.dataset.w)
    span.className = cls[status]
    if (status === 'known') span.removeAttribute('tabindex')
    else span.setAttribute('tabindex', '0')
  })
}

// ── word gestures: click / shift / alt / long-press (event delegation) ──

const LONG_PRESS_MS = 450
let lpTimer = null, lpFired = false

document.addEventListener('click', (e) => {
  if (S.tab !== 'rd') return
  if (lpFired) { lpFired = false; return }
  const span = e.target.closest('#reader-text [data-w]')
  if (!span) return
  const wordId = span.dataset.w
  const status = getStatus(wordId)
  if (status === 'known') return
  const paragraph = span.closest('[data-pid]')
  const pid = paragraph?.dataset.pid ?? null
  const sentence = pid ? mem.paraById.get(pid)?.text ?? '' : ''
  const token = findToken(pid, wordId)
  if (e.altKey || e.metaKey) return openWordCard(wordId, span.textContent, sentence, token)
  if (e.shiftKey) return void vocabMarkKnown(wordId).catch(() => {})
  if (status === 'queued' || status === 'exported') return openWordCard(wordId, span.textContent, sentence, token)
  void vocabEnqueue(wordId, pid, sentence).catch(() => {})
})

document.addEventListener('touchstart', (e) => {
  if (S.tab !== 'rd') return
  const span = e.target.closest('#reader-text [data-w]')
  if (!span) return
  lpFired = false
  lpTimer = setTimeout(() => {
    lpFired = true
    if (getStatus(span.dataset.w) !== 'known') void vocabMarkKnown(span.dataset.w).catch(() => {})
  }, LONG_PRESS_MS)
}, { passive: true })

for (const ev of ['touchend', 'touchcancel', 'touchmove']) {
  document.addEventListener(ev, () => { clearTimeout(lpTimer) }, { passive: true })
}

function findToken(paragraphId, wordId) {
  const p = paragraphId ? mem.paraById.get(paragraphId) : null
  return p?.tokens.find((t) => t.word_id === wordId) ?? null
}

// ═════════════════════════════════════════════════════════════ Wortkarte

function openWordCard(wordId, surfaceForm, sentence, token = null) {
  S.selectedWord = {
    wordId,
    surfaceForm,
    sentence,
    grammaticalRole: token?.grammatical_role ?? null,
    caseLabel: token?.case_label ?? null,
  }
  saveState()
  switchTab('wc')
}

async function renderWordCard() {
  const root = $('#screen-wc')
  const sel = S.selectedWord
  if (!sel) {
    root.innerHTML = `<div class="empty-state">Wähle ein Wort im Leseraum um die Wortkarte zu öffnen.</div>`
    return
  }
  const key = `${sel.wordId ?? sel.surfaceForm}|${sel.sentence.slice(0, 120)}`

  const draw = (annotation, record) => {
    const status = sel.wordId ? getStatus(sel.wordId) : 'new'
    const stateLabel = { queued: 'In der Warteschlange', known: 'Bekannt', exported: 'Exportiert', new: 'Neu · nicht markiert' }[status]
    const genderLabel = record?.gender
      ? { n: 'Neutrum', m: 'Maskulinum', f: 'Femininum' }[record.gender] ?? null : null
    root.innerHTML = `
      <button class="back-btn" id="wc-back">← Zurück zum Text</button>
      <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px">
          <div>
            <div class="word-hero">${esc(sel.surfaceForm)}</div>
            <div class="word-grammar-info">
              ${esc(record?.lemma ?? sel.surfaceForm)}${genderLabel ? ` · ${genderLabel}` : ''}${record?.pos ? ` · ${esc(record.pos)}` : ''}
            </div>
          </div>
          <span class="badge badge-amber" style="white-space:nowrap;flex-shrink:0;margin-left:10px">${stateLabel}</span>
        </div>
      </div>
      <div class="card">
        <div class="card-label">Bedeutung</div>
        ${annotation ? `
          <div style="font-size:15px;margin-bottom:6px;line-height:1.55">${esc(annotation.definition_de)}</div>
          <div style="font-size:13px;color:var(--color-text-secondary);margin-bottom:6px">${esc(annotation.definition_en)}</div>
          ${annotation.etymology ? `<div style="font-size:12px;color:var(--color-text-secondary);font-style:italic">${esc(annotation.etymology)}</div>` : ''}
          ${annotation.source === 'mock' ? `<div style="margin-top:8px;font-size:10px;color:var(--color-text-tertiary)">Offline-Erklärung — Ollama starten für AI.</div>` : ''}`
        : `<div class="skeleton">Lädt Erklärung…</div>`}
      </div>
      <div class="card">
        <div class="card-label">Grammatik in dieser Passage</div>
        <div class="grammar-grid">
          <div class="grammar-cell"><div class="grammar-cell-label">Kasus</div><div class="grammar-cell-value">${esc(sel.caseLabel ?? '—')}</div></div>
          <div class="grammar-cell"><div class="grammar-cell-label">Rolle</div><div class="grammar-cell-value">${esc(sel.grammaticalRole ?? '—')}</div></div>
          <div class="grammar-cell"><div class="grammar-cell-label">Genus</div><div class="grammar-cell-value">${genderLabel ?? '—'}</div></div>
        </div>
      </div>
      ${annotation?.literary_note ? `
      <div class="card">
        <div class="card-label">Literarischer Kontext</div>
        <div class="lit-quote">„${esc(sel.sentence)}“</div>
        <div style="font-size:13px;color:var(--color-text-secondary);line-height:1.65">${esc(annotation.literary_note)}</div>
      </div>` : ''}
      ${annotation?.related_words?.length ? `
      <div class="card">
        <div class="card-label">Verwandte Wörter</div>
        <div style="display:flex;gap:6px;flex-wrap:wrap">
          ${annotation.related_words.map((w) => `<button class="chip" data-related="${esc(w)}">${esc(w)}</button>`).join('')}
        </div>
      </div>` : ''}
      <div class="btn-row" style="flex-wrap:wrap">
        <button class="btn-secondary" id="wc-close">Schließen</button>
        ${sel.wordId ? `
          ${(status === 'queued' || status === 'exported') ? `<button class="btn-secondary" id="wc-remove">Aus Warteschlange entfernen</button>` : ''}
          ${status !== 'known' ? `<button class="btn-secondary" id="wc-known">Als bekannt markieren</button>` : ''}
          ${status !== 'queued' ? `<button class="btn-primary" id="wc-enqueue">In Warteschlange</button>` : ''}` : ''}
      </div>`
    $('#wc-back').onclick = $('#wc-close').onclick = () => { S.selectedWord = null; saveState(); switchTab('rd') }
    $('#wc-remove')?.addEventListener('click', () => sel.wordId && vocabRemove(sel.wordId).then(renderWordCard).catch(() => renderWordCard()))
    $('#wc-known')?.addEventListener('click', () => sel.wordId && vocabMarkKnown(sel.wordId).then(renderWordCard).catch(() => renderWordCard()))
    $('#wc-enqueue')?.addEventListener('click', async () => {
      if (!sel.wordId) return
      try { await vocabEnqueue(sel.wordId, null, sel.sentence) } catch {}
      S.selectedWord = null; saveState(); switchTab('rd')
    })
    $$('[data-related]', root).forEach((b) => b.onclick = () => {
      S.selectedWord = { wordId: null, surfaceForm: b.dataset.related, sentence: sel.sentence, grammaticalRole: null, caseLabel: null }
      saveState(); renderWordCard()
    })
  }

  const cachedAnn = mem.annotation.get(key)
  const cachedWord = sel.wordId ? mem.wordById.get(sel.wordId) : null
  draw(cachedAnn ?? null, cachedWord ?? null)

  try {
    const [annotation, record] = await Promise.all([
      mem.annotation.has(key) ? Promise.resolve(cachedAnn) : api('/annotations/word', {
        method: 'POST',
        body: JSON.stringify({
          word: sel.surfaceForm, sentence: sel.sentence,
          grammatical_role: sel.grammaticalRole, case_label: sel.caseLabel,
        }),
      }),
      sel.wordId
        ? mem.wordById.has(sel.wordId) ? Promise.resolve(cachedWord)
          : api(`/words/${encodeURIComponent(sel.wordId)}`).catch(() => null)
        : Promise.resolve(null),
    ])
    if (annotation) mem.annotation.set(key, annotation)
    if (record) mem.wordById.set(record.id, record)
    if (S.tab === 'wc' && S.selectedWord === sel) draw(annotation ?? null, record ?? null)
  } catch {
    if (S.tab === 'wc' && S.selectedWord === sel && !cachedAnn) {
      root.querySelector('.card:nth-of-type(2)').innerHTML =
        `<div class="card-label">Bedeutung</div><div class="empty-state">Fehler beim Laden der Erklärung.</div>`
    }
  }
}

// ═════════════════════════════════════════════════════════════ TTS (Piper + browser fallback)

const tts = {
  status: 'idle', chunks: [], idx: 0, backend: 'piper', speed: 1.0,
  runId: 0, audio: null, url: null, onActive: null,

  setSpeed(s) { this.speed = s; this.applySpeed() },
  applySpeed() {
    if (this.audio) this.audio.playbackRate = this.speed
  },

  stop() {
    this.runId++
    if (this.audio) { this.audio.pause(); this.audio.src = '' ; this.audio = null }
    if (this.url) { URL.revokeObjectURL(this.url); this.url = null }
    window.speechSynthesis?.cancel()
    this.status = 'idle'; this.idx = 0
    this.onActive?.(null)
    ttsBarRender()
  },

  async play(chunks, startIndex = 0) {
    this.stop()
    const run = ++this.runId
    this.chunks = chunks; this.idx = startIndex; this.status = 'loading'
    for (let i = startIndex; i < chunks.length; i++) {
      if (run !== this.runId) return
      this.idx = i
      this.onActive?.(chunks[i].id)
      ttsBarRender()
      const piperWorked = await this.playPiperChunk(chunks[i].text, run)
      if (run !== this.runId) return
      if (!piperWorked) {
        this.backend = 'browser'
        return this.playBrowser(chunks, i, run)
      }
    }
    if (run !== this.runId) return
    this.status = 'idle'; this.onActive?.(null); ttsBarRender()
  },

  // Returns false when the backend is unavailable (503) so we can fall back.
  async playPiperChunk(text, run) {
    this.backend = 'piper'
    try {
      const res = await fetch(`${BASE}/voice/tts`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, speed: this.speed }),
      })
      if (res.status === 503) return false
      if (!res.ok) throw new Error(`TTS ${res.status}`)
      const blob = await res.blob()
      if (run !== this.runId) return true
      await new Promise((resolve) => {
        const audio = new Audio(URL.createObjectURL(blob))
        this.audio = audio; this.url = audio.src
        this.status = 'playing'; ttsBarRender()
        audio.playbackRate = this.speed
        audio.onended = resolve
        audio.onerror = resolve
        audio.play().catch(resolve)
      })
      if (this.url) { URL.revokeObjectURL(this.url); this.url = null }
      this.audio = null
      return true
    } catch {
      // Network-level failure mid-stream: stop cleanly rather than fallback.
      if (run !== this.runId) return true
      this.status = 'error'; ttsBarRender()
      return true
    }
  },

  playBrowser(list, startIndex, run) {
    if (!('speechSynthesis' in window)) { this.status = 'error'; ttsBarRender(); return }
    const speak = (i) => {
      if (run !== this.runId) return
      if (i >= list.length) { this.status = 'idle'; this.onActive?.(null); ttsBarRender(); return }
      this.idx = i; this.status = 'playing'
      this.onActive?.(list[i].id); ttsBarRender()
      const u = new SpeechSynthesisUtterance(list[i].text)
      const voices = speechSynthesis.getVoices()
      const de = voices.find((v) => v.lang.startsWith('de')) ?? voices.find((v) => v.default) ?? voices[0]
      u.lang = de?.lang ?? 'de-DE'
      if (de) u.voice = de
      u.rate = this.speed
      u.onend = () => speak(i + 1)
      u.onerror = () => { this.status = 'error'; ttsBarRender() }
      speechSynthesis.speak(u)
    }
    speak(startIndex)
  },

  pause() {
    if (this.audio) { this.audio.pause(); this.status = 'paused' }
    else if ('speechSynthesis' in window && speechSynthesis.speaking) { speechSynthesis.pause(); this.status = 'paused' }
    ttsBarRender()
  },
  resume() {
    if (this.audio) { this.audio.play().catch(() => {}); this.status = 'playing' }
    else if ('speechSynthesis' in window) { speechSynthesis.resume(); this.status = 'playing' }
    ttsBarRender()
  },

  mountBar(slot, chunks) {
    this.barSlot = slot
    this.pageChunks = chunks
    ttsBarRender()
  },
}

function ttsBarRender() {
  const slot = tts.barSlot
  if (!slot) return
  const playing = tts.status === 'playing' || tts.status === 'loading'
  const total = Math.max(tts.chunks.length ?? tts.pageChunks?.length ?? 0, 1)
  const pos = tts.status === 'idle' ? 0 : Math.min(tts.idx + (playing ? 1 : 0), total)
  const label = tts.backend === 'piper' ? 'Piper · de_DE-thorsten-high' : 'Browser TTS (Fallback)'
  slot.innerHTML = `
    <div class="tts-bar">
      <button class="tts-play" id="tts-toggle" ${tts.pageChunks?.length ? '' : 'disabled'}
              title="${playing ? 'Pause' : 'Vorlesen'}">${playing ? '❚❚' : '▶'}</button>
      <div class="tts-info">
        <div class="tts-label">${label}${tts.status !== 'idle' ? ` · Absatz ${Math.min(tts.idx + 1, total)}/${total}` : ''}</div>
        <div class="tts-progress"><div class="tts-fill" style="width:${Math.round((pos / total) * 100)}%"></div></div>
      </div>
      <div class="speed-controls" role="group" aria-label="Vorlese-Tempo">
        ${[0.8, 1.0, 1.2].map((s) =>
          `<button class="${s === S.readerPrefs.ttsSpeed ? 'speed-active' : 'speed-muted'}" data-speed="${s}">${s.toFixed(1)}×</button>`).join('')}
      </div>
    </div>`
  $('#tts-toggle').onclick = () => {
    if (tts.status === 'idle' || tts.status === 'error') tts.play(tts.pageChunks ?? [], 0)
    else if (playing) tts.pause()
    else tts.resume()
  }
  $$('[data-speed]', slot).forEach((b) => b.onclick = () => {
    S.readerPrefs.ttsSpeed = Number(b.dataset.speed)
    tts.setSpeed(S.readerPrefs.ttsSpeed)
    saveState(); ttsBarRender()
  })
}

// Highlight the paragraph currently being read.
tts.onActive = (id) => {
  $$('#reader-text p').forEach((p) => p.classList.toggle('tts-active', p.dataset.pid === id))
}

// ═════════════════════════════════════════════════════════════ Stimme (chat + mic + AI settings)

const QUICK_CHIPS = [
  'Was bedeutet das auf Englisch?',
  'Erkläre die Grammatik',
  'Historischer Kontext',
  'Schwierige Stellen',
]

const voice = { initialised: false, paragraph: null }

async function initVoice() {
  const root = $('#screen-vx')
  if (!voice.initialised) {
    voice.initialised = true
    root.addEventListener('click', (e) => {
      if (e.target.id === 'vx-back') switchTab('rd')
      const chip = e.target.closest('[data-chip]')
      if (chip) { $('#vx-input').value = chip.dataset.chip; sendChat() }
    })
    root.addEventListener('click', (e) => {
      if (e.target.closest('#vx-send')) sendChat()
      if (e.target.closest('#vx-clear')) { mem.chat.delete(S.currentParagraphId); drawChat() }
      if (e.target.closest('#vx-mic')) handleMic()
    })
    root.addEventListener('keydown', (e) => {
      if (e.target.id === 'vx-input' && e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat() }
    })
  }
  root.innerHTML = `
    <button class="back-btn" id="vx-back">← Zurück zum Text</button>
    <div id="ai-settings-slot"></div>
    <div class="passage-context">
      <div class="card-label">Aktuelle Passage</div>
      <div id="vx-passage" style="font-family:var(--font-serif);font-size:14px;color:var(--color-text-secondary);line-height:1.75">
        <span class="skeleton">Keine Passage ausgewählt.</span>
      </div>
    </div>
    <div class="chat-history" id="vx-history"></div>
    <div class="voice-input-row">
      <button class="mic-btn" id="vx-mic" title="Frage sprechen">🎤</button>
      <input id="vx-input" type="text" placeholder="Frage zum Text…" style="flex:1">
      <button class="btn-primary" id="vx-send">Fragen</button>
      <button class="btn-secondary" id="vx-clear" title="Verlauf löschen">✕</button>
    </div>`
  drawChat()
  renderAiSettings($('#ai-settings-slot'))
  if (S.currentParagraphId) {
    api(`/corpus/paragraphs/${encodeURIComponent(S.currentParagraphId)}`)
      .then((p) => { voice.paragraph = p; $('#vx-passage').textContent = `„${p.text}“` })
      .catch(() => {})
  }
}

function drawChat(draft = '', status = 'idle') {
  const history = mem.chat.get(S.currentParagraphId) ?? []
  const el = $('#vx-history')
  if (!el) return
  el.innerHTML = history.map((t) =>
    `<div class="${t.role === 'user' ? 'chat-user' : 'chat-ai'}">
       <div class="${t.role === 'user' ? 'chat-bubble-user' : 'chat-bubble-ai'}">${esc(t.content)}</div>
     </div>`).join('') +
    (draft ? `<div class="chat-ai"><div class="chat-bubble-ai">${esc(draft)}<span class="chat-caret"></span></div></div>` : '') +
    (history.length === 0 && status === 'idle' && !draft
      ? `<div class="chat-ai"><div class="chat-bubble-ai">Frage etwas zum Text — Kontext und Nachbarn werden mitgeschickt.</div></div>` : '')
  const chips = document.createElement('div')
  chips.className = 'chips-row'
  chips.innerHTML = QUICK_CHIPS.map((c) => `<button class="chip" data-chip="${esc(c)}">${esc(c)}</button>`).join('')
  el.prepend(chips)
  el.scrollTop = el.scrollHeight
}

let chatBusy = false

async function sendChat() {
  const input = $('#vx-input')
  const question = input.value.trim()
  if (!question || chatBusy || !S.currentParagraphId) return
  chatBusy = true
  input.value = ''
  const pid = S.currentParagraphId
  const history = mem.chat.get(pid) ?? []
  history.push({ role: 'user', content: question })
  let accumulated = ''
  drawChat('', 'streaming')
  try {
    const res = await fetch(`${BASE}/chat/message`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify({ question, paragraph_id: pid, history: history.slice(0, -1) }),
    })
    if (!res.ok || !res.body) throw new Error(`Chat ${res.status}`)
    const reader = res.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    const handle = (block) => {
      let event = 'message'
      const data = []
      for (const line of block.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim()
        else if (line.startsWith('data:')) data.push(line.slice(5).trim())
      }
      if (data.length === 0) return true
      const raw = data.join('\n')
      if (event === 'token') {
        try { accumulated += JSON.parse(raw).delta ?? '' } catch {}
        drawChat(accumulated, 'streaming')
      } else if (event === 'error') {
        let detail = raw
        try { detail = JSON.parse(raw).detail || 'stream error' } catch {}
        accumulated += `\n(Fehler: ${detail})`
        drawChat(accumulated, 'error')
      } else if (event === 'done') return false
      return true
    }
    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let idx
      while ((idx = buffer.indexOf('\n\n')) !== -1) {
        const block = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 2)
        if (!handle(block)) { reader.cancel().catch(() => {}); break }
      }
    }
  } catch (err) {
    accumulated += `\n(Fehler: ${err.message})`
  }
  if (accumulated.trim()) history.push({ role: 'assistant', content: accumulated })
  mem.chat.set(pid, history)
  chatBusy = false
  drawChat()
  if (S.readerPrefs.ttsAutoPlay && accumulated.trim()) {
    tts.play([{ id: `answer-${history.length}`, text: accumulated }])
  }
}

// ── mic (MediaRecorder → /voice/stt) ────────────────────────────────────

let recorder = null, recStream = null, recChunks = []

async function handleMic() {
  const btn = $('#vx-mic')
  if (recorder && recorder.state === 'recording') {
    const transcript = await new Promise((resolve) => {
      recorder.onstop = async () => {
        recStream.getTracks().forEach((t) => t.stop())
        const blob = new Blob(recChunks, { type: recorder.mimeType || 'audio/webm' })
        try {
          const form = new FormData()
          form.append('audio', blob, blob.type.includes('webm') ? 'recording.webm' : 'recording.ogg')
          const res = await fetch(`${BASE}/voice/stt`, { method: 'POST', body: form })
          if (res.status === 503) { alert('Spracherkennung nicht installiert (faster-whisper).'); return resolve(null) }
          if (!res.ok) throw new Error(`STT ${res.status}`)
          resolve((await res.json()).transcript)
        } catch (err) { alert(`STT fehlgeschlagen: ${err.message}`); resolve(null) }
      }
      recorder.stop()
    })
    if (transcript) $('#vx-input').value = transcript
    btn.classList.remove('recording')
    return
  }
  try {
    recStream = await navigator.mediaDevices.getUserMedia({ audio: true })
    recChunks = []
    const mime = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
      .find((m) => MediaRecorder.isTypeSupported?.(m)) || ''
    recorder = new MediaRecorder(recStream, mime ? { mimeType: mime } : undefined)
    recorder.ondataavailable = (e) => e.data.size && recChunks.push(e.data)
    recorder.start()
    btn.classList.add('recording')
  } catch { alert('Mikrofon nicht verfügbar.') }
}

// ── AI settings (runtime LLM options) ──────────────────────────────────

async function renderAiSettings(slot) {
  slot.innerHTML = `<div class="ai-settings-panel skeleton">Lädt KI-Einstellungen…</div>`
  let data
  try { data = await api('/admin/ollama') } catch { slot.innerHTML = ''; return }
  const { options, ollama } = data
  const draw = (opts, models) => {
    slot.innerHTML = `
      <details class="ai-settings-panel">
        <summary class="ai-settings-trigger">
          <span class="ai-status-dot" style="background:${ollama.reachable && ollama.configured_available ? 'var(--teal-400)' : 'var(--amber-400)'}"></span>
          ${esc(opts.model)}
          ${ollama.reachable && !ollama.configured_available ? ' — Modell nicht installiert' : ''}
        </summary>
        <div class="ai-settings">
          <div class="ai-settings-row">
            <span class="ai-settings-label">Modell</span>
            <select id="ai-model" class="reader-nav-select" style="flex:1">
              ${(models.length ? models : [opts.model]).map((m) =>
                `<option value="${esc(m)}" ${m === opts.model ? 'selected' : ''}>${esc(m)}</option>`).join('')}
            </select>
          </div>
          <div class="ai-settings-row">
            <span class="ai-settings-label">Denk-Modus</span>
            <input type="checkbox" id="ai-think" class="ai-toggle" ${opts.think ? 'checked' : ''}>
          </div>
          <div class="ai-settings-row">
            <span class="ai-settings-label">Temperatur <span class="ai-settings-value" id="ai-temp-value">${opts.temperature.toFixed(2)}</span></span>
            <input type="range" id="ai-temp" class="ai-slider" min="0" max="2" step="0.05" value="${opts.temperature}">
          </div>
          ${ollama.reachable ? '' : `<div class="ai-settings-hint">Ollama nicht erreichbar — Chat nutzt Offline-Antworten.</div>`}
        </div>
      </details>`
    const put = (patch) => api('/admin/ollama', { method: 'PUT', body: JSON.stringify(patch) })
      .then((r) => draw(r.options, models))
      .catch(() => {})
    $('#ai-model').onchange = (e) => put({ model: e.target.value })
    $('#ai-think').onchange = (e) => put({ think: e.target.checked })
    $('#ai-temp').oninput = (e) => { $('#ai-temp-value').textContent = Number(e.target.value).toFixed(2) }
    $('#ai-temp').onchange = (e) => put({ temperature: Number(e.target.value) })
  }
  draw(options, ollama.models)
}

// ═════════════════════════════════════════════════════════════ Export

const exportState = { entries: [], drafts: {}, annotating: {}, busy: false, message: null, initialised: false }

function autoQuestion(e) {
  if (e.question_override?.trim()) return e.question_override.trim()
  if (e.pos === 'NOUN') {
    const art = { m: 'der', f: 'die', n: 'das' }[e.gender]
    return art ? `${art} ${e.lemma} (${e.gender})` : `${e.lemma} (n)`
  }
  if (e.pos === 'VERB') return `${e.lemma} (v)`
  if (e.pos === 'ADJ') return `${e.lemma} (adj)`
  return e.lemma
}

function autoAnswer(e) {
  return [e.definition_de?.trim(), e.definition_en?.trim()].filter(Boolean).join('\n')
}

async function initExport() {
  const root = $('#screen-ex')
  if (!exportState.initialised) {
    exportState.initialised = true
    root.addEventListener('click', exportClick)
    root.addEventListener('focusout', exportBlur)
  }
  await loadExport()
}

async function loadExport() {
  const root = $('#screen-ex')
  root.innerHTML = `<div class="skeleton">Lädt Warteschlange…</div>`
  try {
    exportState.entries = await api('/vocab/queue?status=queued')
    exportState.drafts = Object.fromEntries(exportState.entries.map((e) => [e.word_id, {
      question: e.question_override ?? '', answer: e.answer_override ?? '', extra_tags: e.extra_tags ?? '',
    }]))
  } catch (err) {
    root.innerHTML = `<div class="empty-state">Fehler: ${esc(err.message)}</div>`
    return
  }
  drawExport()
}

function drawExport() {
  const root = $('#screen-ex')
  const { entries } = exportState
  const needsAnnotation = entries.filter((e) => !autoAnswer(e)).length
  const annotatable = entries.filter((e) => {
    if (e.answer_override?.trim()) return false
    const hasDe = Boolean(e.definition_de?.trim())
    const hasEn = Boolean(e.definition_en?.trim())
    return !hasDe || !hasEn
  })
  exportState.message = null
  root.innerHTML = `
    ${exportState.message ? `<div class="card" style="margin-bottom:10px">${esc(exportState.message)}</div>` : ''}
    <div class="card" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
      <div>
        <div class="card-label">Export-Warteschlange</div>
        <div style="font-size:18px;font-weight:600">${entries.length} Wörter zum Export
          ${needsAnnotation > 0 ? `<span style="margin-left:10px;font-size:13px;color:var(--amber-800);font-weight:400">· ${needsAnnotation} brauchen Annotation</span>` : ''}
        </div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end">
        <button class="btn-secondary" id="ex-refresh" ${exportState.busy ? 'disabled' : ''}>Aktualisieren</button>
        <button class="btn-secondary" id="ex-bulk" ${annotatable.length === 0 || exportState.busy ? 'disabled' : ''}>
          Alle annotieren${annotatable.length ? ` (${annotatable.length})` : ''}</button>
        <button class="btn-primary" id="ex-export" ${needsAnnotation > 0 || exportState.busy ? 'disabled' : ''}>
          ${entries.length > 0 ? `Export ${entries.length} Karten als CSV` : 'Export CSV'}</button>
      </div>
    </div>
    ${entries.length === 0
      ? `<div class="empty-state">Noch keine Wörter in der Warteschlange. Klicke Wörter im Leseraum, um sie hier zu sammeln.</div>`
      : `<div style="display:flex;flex-direction:column;gap:10px">${entries.map(exportRowHtml).join('')}</div>`}`
}

function exportRowHtml(e) {
  const d = exportState.drafts[e.word_id] ?? { question: '', answer: '', extra_tags: '' }
  const ans = autoAnswer(e)
  return `
    <div class="card" data-row="${esc(e.word_id)}" style="display:flex;flex-direction:column;gap:8px">
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:8px;flex-wrap:wrap">
        <div>
          <strong style="font-size:16px">${esc(autoQuestion(e))}</strong>
          <span style="margin-left:8px;font-size:12px;color:var(--color-text-secondary)">${esc(e.pos ?? '—')}</span>
        </div>
        <div style="display:flex;gap:6px">
          ${!ans ? `<button class="btn-primary" data-annotate="${esc(e.word_id)}" ${exportState.annotating[e.word_id] ? 'disabled' : ''}>
            ${exportState.annotating[e.word_id] ? 'Annotiere…' : 'Annotieren'}</button>` : ''}
          <button class="btn-secondary" data-remove="${esc(e.word_id)}">Entfernen</button>
        </div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <input data-field="question" data-id="${esc(e.word_id)}" value="${esc(d.question)}"
               placeholder="${esc(autoQuestion(e))}" title="Frage (leer = automatisch)">
        <input data-field="extra_tags" data-id="${esc(e.word_id)}" value="${esc(d.extra_tags)}" placeholder="Tags, Komma-getrennt">
      </div>
      <textarea data-field="answer" data-id="${esc(e.word_id)}" rows="2" style="width:100%;box-sizing:border-box"
                placeholder="${esc(ans || 'Antwort (Annotieren für AI-Definition)')}">${esc(d.answer)}</textarea>
      <div style="font-size:11px;color:var(--color-text-tertiary)">${esc(ans).replace(/\n/g, ' · ')}</div>
    </div>`
}

function exportClick(e) {
  const annotate = e.target.closest('[data-annotate]')
  if (annotate) return annotateEntry(annotate.dataset.annotate)
  const remove = e.target.closest('[data-remove]')
  if (remove) return removeEntry(remove.dataset.remove)
  if (e.target.id === 'ex-refresh') return loadExport()
  if (e.target.id === 'ex-bulk') return bulkAnnotate()
  if (e.target.id === 'ex-export') return exportCsv()
}

async function exportBlur(e) {
  const input = e.target.closest('[data-field]')
  if (!input) return
  const id = input.dataset.id
  const field = input.dataset.field
  exportState.drafts[id] ??= { question: '', answer: '', extra_tags: '' }
  exportState.drafts[id][field] = input.value
  const entry = exportState.entries.find((x) => x.word_id === id)
  if (!entry) return
  const d = exportState.drafts[id]
  const body = {
    question: d.question.trim() || null,
    answer: d.answer.trim() || null,
    extra_tags: d.extra_tags.trim() || null,
  }
  if ((body.question ?? '') === (entry.question_override ?? '')
    && (body.answer ?? '') === (entry.answer_override ?? '')
    && (body.extra_tags ?? '') === (entry.extra_tags ?? '')) return
  try {
    const updated = await api(`/vocab/queue/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(body) })
    exportState.entries = exportState.entries.map((x) => x.word_id === updated.word_id ? updated : x)
  } catch (err) { exportState.message = `Speichern fehlgeschlagen: ${err.message}` }
  drawExport()
}

async function annotateEntry(id) {
  exportState.annotating[id] = true
  drawExport()
  try {
    const updated = await api(`/vocab/queue/${encodeURIComponent(id)}/annotate`, { method: 'POST' })
    exportState.entries = exportState.entries.map((x) => x.word_id === updated.word_id ? updated : x)
    exportState.drafts[id] = { question: updated.question_override ?? '', answer: updated.answer_override ?? '', extra_tags: updated.extra_tags ?? '' }
  } catch (err) { exportState.message = `Annotation fehlgeschlagen: ${err.message}` }
  delete exportState.annotating[id]
  drawExport()
}

async function bulkAnnotate() {
  const targets = exportState.entries.filter((e) => {
    if (e.answer_override?.trim()) return false
    const hasDe = Boolean(e.definition_de?.trim())
    const hasEn = Boolean(e.definition_en?.trim())
    return !hasDe || !hasEn
  })
  if (targets.length === 0 || exportState.busy) return
  exportState.busy = true
  let failed = 0
  for (let i = 0; i < targets.length; i++) {
    const e = targets[i]
    exportState.annotating[e.word_id] = true
    exportState.message = `Annotiere ${i + 1}/${targets.length}…`
    drawExport()
    try {
      const updated = await api(`/vocab/queue/${encodeURIComponent(e.word_id)}/annotate`, { method: 'POST' })
      exportState.entries = exportState.entries.map((x) => x.word_id === updated.word_id ? updated : x)
      exportState.drafts[e.word_id] = { question: updated.question_override ?? '', answer: updated.answer_override ?? '', extra_tags: updated.extra_tags ?? '' }
    } catch { failed++ }
    delete exportState.annotating[e.word_id]
  }
  exportState.busy = false
  exportState.message = failed === 0
    ? `${targets.length} Wörter annotiert.`
    : `${targets.length - failed} von ${targets.length} annotiert (${failed} fehlgeschlagen).`
  drawExport()
}

async function removeEntry(id) {
  try {
    await vocabRemove(id)
    exportState.entries = exportState.entries.filter((e) => e.word_id !== id)
  } catch (err) { exportState.message = `Entfernen fehlgeschlagen: ${err.message}` }
  drawExport()
}

async function exportCsv() {
  const ids = exportState.entries.map((e) => e.word_id)
  exportState.busy = true
  drawExport()
  try {
    const res = await fetch(`${BASE}/vocab/export`, { method: 'POST' })
    if (res.status === 204) {
      exportState.message = 'Keine Karten im Export.'
    } else if (res.ok) {
      const blob = await res.blob()
      const disposition = res.headers.get('content-disposition') || ''
      const match = disposition.match(/filename\*?=(?:UTF-8'')?"?([^";]+)"?/i)
      const a = document.createElement('a')
      a.href = URL.createObjectURL(blob)
      a.download = match?.[1]?.trim() || `zenkai-vocab-${new Date().toISOString().slice(0, 10)}.csv`
      document.body.appendChild(a); a.click(); a.remove()
      setTimeout(() => URL.revokeObjectURL(a.href), 0)
      for (const id of ids) S.vocabStatus[id] = 'exported'
      exportState.entries = []
      exportState.drafts = {}
      exportState.message = `${ids.length} Karten exportiert.`
      saveState(); refreshCounts()
    } else {
      exportState.message = `Export fehlgeschlagen: ${res.status}`
    }
  } catch (err) { exportState.message = `Export fehlgeschlagen: ${err.message}` }
  exportState.busy = false
  drawExport()
}

// ── boot ────────────────────────────────────────────────────────────────

loadState()
renderTabs()
switchTab(['lib', 'rd', 'wc', 'ex', 'vx'].includes(S.tab) ? S.tab : 'lib')
