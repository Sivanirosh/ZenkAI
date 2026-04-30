// Shared TypeScript interfaces — mirrors backend/models/pydantic_models.py.

export type TabId = 'lib' | 'rd' | 'wc' | 'vx' | 'ex'

export interface Work {
  id: string
  title: string
  author: string
  epoch: string | null
  year: number | null
  gutenberg_id: number | null
  language: string
  spine_color: string | null
  epoch_color: string | null
}

export interface WorkProgress {
  work_id: string
  known_pct: number
  learning_pct: number
  new_pct: number
  total_words: number
  known_words: number
}

export interface WorkWithProgress extends Work {
  progress: WorkProgress
}

export interface Chapter {
  work_id: string
  chapter: number
  paragraph_count: number
  first_paragraph_id: string
}

export interface WordToken {
  surface_form: string
  word_id: string | null
  lemma: string | null
  pos: string | null
  case_label: string | null
  grammatical_role: string | null
  familiarity: number
  char_start: number
  char_end: number
}

export interface Paragraph {
  id: string
  work_id: string
  chapter: number
  position: number
  text: string
  word_count: number
}

export interface ParagraphWithTokens extends Paragraph {
  tokens: WordToken[]
}

export interface WordState {
  word_id: string
  familiarity: number
  ease_factor: number
  interval: number
  next_review: string | null
  seen_count: number
  last_seen: string | null
}

export interface Word {
  id: string
  lemma: string
  pos: string | null
  gender: string | null
  plural_form: string | null
  etymology: string | null
  definition_de: string | null
  definition_en: string | null
}

export interface WordWithState extends Word {
  state: WordState
}

export interface WordAnnotation {
  definition_de: string
  definition_en: string
  literary_note: string | null
  etymology: string | null
  related_words: string[]
  source: 'ollama' | 'mock'
}

export interface ProgressSummary {
  total_words_tracked: number
  known: number
  learning: number
  new: number
  streak_days: number
}

export interface ChatTurn {
  role: 'user' | 'assistant'
  content: string
}

export interface ChatRequest {
  question: string
  paragraph_id: string | null
  history: ChatTurn[]
}

export interface ChatTokenEvent {
  delta: string
}

export interface SttResponse {
  transcript: string
  confidence: number
  language: string
  duration_s: number
}

export interface ServiceUnavailable {
  detail: string
  install_hint: string
}

// ── Vocabulary harvester (decision 0001) ──────────────────────

export type VocabStatus = 'queued' | 'known' | 'exported'

export interface VocabEntry {
  word_id: string
  status: VocabStatus
  lemma: string
  pos: string | null
  gender: string | null
  definition_de: string | null
  definition_en: string | null
  source_paragraph: string | null
  source_sentence: string | null
  added_at: string | null
  exported_at: string | null
  question_override: string | null
  answer_override: string | null
  extra_tags: string | null
  needs_annotation: boolean
}

export interface VocabCounts {
  queued: number
  known: number
  exported: number
}

export interface VocabStatusMap {
  statuses: Record<string, VocabStatus>
}
