// Shared TypeScript interfaces — mirrors backend/models/pydantic_models.py.

export type TabId =
  | 'on'
  | 'at'
  | 'al'
  | 'kv'
  | 'cap'
  | 'lib'
  | 'rd'
  | 'wc'
  | 'vx'
  | 'ex'

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

// ── Mira agent (PIVOT_ROADMAP §6.3 + §A.6/A.13) ────────────────

export interface MiraTurnRequest {
  current_screen?: string
  time_budget_min?: number | null
  current_paragraph_id?: string | null
  work_id?: string | null
  user_message?: string | null
  session_id?: string | null
  goal_text?: string | null
  extras?: Record<string, unknown>
}

export interface ToolIntent {
  name: string
  args: Record<string, unknown>
  call_id?: string
}

export interface ToolResult {
  name: string
  call_id?: string
  // Tool-specific JSON; we leave it untyped here so each consumer can
  // narrow on the `name` discriminator.
  [key: string]: unknown
}

export interface ToolError {
  tool: string
  code: string
  message: string
  detail?: unknown
}

export interface MiraTurnDone {
  turn_ms?: number
  session_id?: string | null
  reason?: string
}

export type MiraEvent =
  | { kind: 'thought'; data: string }
  | { kind: 'tool_intent'; data: ToolIntent }
  | { kind: 'tool_result'; data: ToolResult }
  | { kind: 'tool_error'; data: ToolError }
  | { kind: 'error'; data: { message: string; detail?: unknown } }
  | { kind: 'done'; data: MiraTurnDone }

export interface AtriumDoor {
  id: string
  label: string
  subtitle: string
}

export interface AtriumMastery {
  competency_id: string
  confidence: number
  variance: number
  label: string
  cefr?: string | null
}

export interface AtriumPayload {
  greeting: string
  today_plan: string[]
  doors: AtriumDoor[]
  mastery_top: AtriumMastery[]
  generated_at: string
}

// ── Onboarding (PIVOT_ROADMAP §B.1) ─────────────────────────────

export type GoalDomain = 'medical' | 'academic' | 'daily' | 'other'
export type TargetCefr = 'A2' | 'B1' | 'B2' | 'C1'

export interface ParsedGoal {
  domain: GoalDomain
  target_cefr: TargetCefr
  deadline_iso: string | null
  scenarios: string[]
  motivations: string[]
  language_profile: {
    native: string
    current_cefr: string
  }
}

export interface OnboardingState {
  has_goal: boolean
  latest_goal_id: string | null
  latest_plan_id: string | null
  horizon?: string | null
  raw_text?: string
  parsed?: ParsedGoal | null
  created_at?: string | null
}

export interface PlanWeek {
  week_index: number
  title: string
  competencies: string[]
  capabilities: string[]
  doors: string[]
}

export interface PlanDistrict {
  id: string
  label: string
  competencies: string[]
  prerequisites: string[]
  week_index: number
  status: 'mastered' | 'current' | 'queued' | 'locked'
}

export interface CurriculumPlan {
  horizon: string
  weeks: PlanWeek[]
  districts: PlanDistrict[]
  edges: [string, string][]
  rationale: string
}

export interface OnboardingGoalResponse {
  goal_id: string
  parsed: ParsedGoal
  plan_id: string
  horizon: string
  plan: CurriculumPlan
}

// ── Atlas (PIVOT_ROADMAP §B.4) ──────────────────────────────────

export interface AtlasGoal {
  id: string
  raw_text: string
  parsed: ParsedGoal | Record<string, unknown>
}

export interface AtlasCompetency {
  competency_id: string
  label: string
  cefr: string | null
  confidence: number
  variance: number
  evidence_count: number
}

export interface AtlasDistrict {
  id: string
  label: string
  week_index: number
  status: 'mastered' | 'current' | 'queued' | 'locked'
  competencies: AtlasCompetency[]
  prerequisites: string[]
  confidence: number
  capability_sentence: string | null
}

export interface AtlasPayload {
  goal: AtlasGoal
  plan_id: string
  horizon: string
  districts: AtlasDistrict[]
  edges: [string, string][]
  current_route: string[]
  rationale: string
  generated_at: string
}

// ── Konversation (PIVOT_ROADMAP §B.5/§B.17) ────────────────────

export interface KonversationTranscript {
  id: string
  session_id: string
  role: 'user' | 'assistant' | 'system'
  text: string
  occurred_at: string
  competency_id: string | null
  scenario: string | null
  audio_seconds: number | null
  stt_engine: string | null
  confidence: number | null
  extras?: Record<string, unknown>
}

export interface KonversationTurnResponse {
  user_transcript: KonversationTranscript
  assistant_reply: KonversationTranscript
  duration_ms: number
  stt_engine: string
  llm_model: string
}

export interface KonversationRecentResponse {
  count: number
  transcripts: KonversationTranscript[]
}

// ── Pronunciation (PIVOT_ROADMAP §B.8) ─────────────────────────

export interface PronunciationSegment {
  word: string
  score: number
  hint: string | null
}

export interface PronunciationScore {
  overall: number | null
  segments: PronunciationSegment[]
  reference_text: string
  reference_audio_path: string | null
  critique: string | null
  engine: 'librosa' | 'skeleton' | string
  duration_ms: number
  metadata?: Record<string, unknown>
}

// ── Capture (PIVOT_ROADMAP §B.6/§B.7/§B.18) ────────────────────

export type CaptureSurfaceKind = 'text' | 'sign' | 'menu' | 'object'

export interface CaptureWord {
  surface: string
  lemma: string | null
  pos: string | null
  gender: string | null
  definition_de: string | null
  definition_en: string | null
}

export interface CaptureResultPayload {
  id: string
  sha256: string
  image_path: string
  surface_kind: CaptureSurfaceKind | string
  transcript: string
  words: CaptureWord[]
  advice: string
  engine: string
  model: string | null
  captured_at: string
  duration_ms: number
}

export interface CaptureRecentRow {
  id: string
  sha256: string
  session_id: string
  surface_kind: string
  target_competency_id: string | null
  image_path: string
  transcript_head: string
  engine: string
  model: string | null
  captured_at: string
}

export interface CaptureRecentResponse {
  count: number
  captures: CaptureRecentRow[]
}
