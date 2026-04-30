-- Lesekamerad DuckDB schema (source of truth).
-- Migrations are append-only ALTER TABLE statements at the bottom of this file.

CREATE TABLE IF NOT EXISTS works (
    id           VARCHAR PRIMARY KEY,
    title        VARCHAR NOT NULL,
    author       VARCHAR NOT NULL,
    epoch        VARCHAR,
    year         INTEGER,
    gutenberg_id INTEGER,
    language     VARCHAR DEFAULT 'de',
    spine_color  VARCHAR,
    epoch_color  VARCHAR
);

CREATE TABLE IF NOT EXISTS paragraphs (
    id         VARCHAR PRIMARY KEY,
    work_id    VARCHAR REFERENCES works(id),
    chapter    INTEGER,
    position   INTEGER,
    text       TEXT NOT NULL,
    word_count INTEGER
);

CREATE TABLE IF NOT EXISTS words (
    id            VARCHAR PRIMARY KEY,
    lemma         VARCHAR NOT NULL,
    pos           VARCHAR,
    gender        VARCHAR,
    plural_form   VARCHAR,
    etymology     TEXT,
    definition_de TEXT,
    definition_en TEXT
);

CREATE TABLE IF NOT EXISTS word_occurrences (
    id               VARCHAR PRIMARY KEY,
    paragraph_id     VARCHAR REFERENCES paragraphs(id),
    word_id          VARCHAR REFERENCES words(id),
    surface_form     VARCHAR NOT NULL,
    position         INTEGER,
    case_label       VARCHAR,
    grammatical_role VARCHAR
);

CREATE TABLE IF NOT EXISTS word_states (
    word_id     VARCHAR PRIMARY KEY REFERENCES words(id),
    familiarity INTEGER DEFAULT 0,
    ease_factor FLOAT DEFAULT 2.5,
    interval    INTEGER DEFAULT 1,
    next_review TIMESTAMP,
    seen_count  INTEGER DEFAULT 0,
    last_seen   TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sessions (
    id                VARCHAR PRIMARY KEY,
    work_id           VARCHAR REFERENCES works(id),
    started_at        TIMESTAMP DEFAULT now(),
    ended_at          TIMESTAMP,
    paragraphs_read   INTEGER DEFAULT 0,
    words_encountered INTEGER DEFAULT 0,
    words_promoted    INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_paragraphs_work_chapter
    ON paragraphs(work_id, chapter, position);
CREATE INDEX IF NOT EXISTS idx_occurrences_paragraph
    ON word_occurrences(paragraph_id, position);
CREATE INDEX IF NOT EXISTS idx_occurrences_word
    ON word_occurrences(word_id);

-- Migrations (append below, never edit above). Each ALTER is idempotent.
-- Example:
-- ALTER TABLE words ADD COLUMN IF NOT EXISTS frequency_band INTEGER;

-- 2026-04: per-word character offsets in paragraphs.text. These make the
-- reader render with correct punctuation (paragraphs.text is the single
-- source of truth) and enable O(N) offset-based slicing without re-running
-- spaCy at serve time.
ALTER TABLE word_occurrences ADD COLUMN IF NOT EXISTS char_start INTEGER;
ALTER TABLE word_occurrences ADD COLUMN IF NOT EXISTS char_end   INTEGER;
CREATE INDEX IF NOT EXISTS idx_occurrences_paragraph_start
    ON word_occurrences(paragraph_id, char_start);

-- 2026-04: vocabulary harvester. External SRS owns scheduling (decision
-- 0001), so word_states SRS columns (ease_factor, interval, next_review,
-- seen_count) are deprecated and preserved for one migration cycle.
-- vocab_queue is the new authoritative state for the reader UI.
CREATE TABLE IF NOT EXISTS vocab_queue (
    word_id            VARCHAR PRIMARY KEY REFERENCES words(id),
    status             VARCHAR NOT NULL,            -- 'queued' | 'known' | 'exported'
    source_paragraph   VARCHAR REFERENCES paragraphs(id),
    source_sentence    TEXT,
    added_at           TIMESTAMP DEFAULT now(),
    exported_at        TIMESTAMP,
    question_override  TEXT,
    answer_override    TEXT,
    extra_tags         TEXT
);
CREATE INDEX IF NOT EXISTS idx_vocab_queue_status ON vocab_queue(status);

-- 2026-04-30: Mira agent (PIVOT_ROADMAP.md §6.1, §7).
-- Four tables introduced together: the goal the learner is chasing, the
-- competency taxonomy Mira reasons over, the per-competency Bayesian
-- mastery state, and the immutable evidence ledger that drives mastery
-- updates. Schema lives here so init_schema() picks it up at startup;
-- the Python seeding lives in backend/mastery/competencies.py.

CREATE TABLE IF NOT EXISTS goals (
    id          VARCHAR PRIMARY KEY,
    raw_text    TEXT NOT NULL,           -- learner's free-text answer to "why are you here?"
    parsed      JSON,                    -- {domain, deadline, target_cefr, scenarios[]}
    created_at  TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS competencies (
    id          VARCHAR PRIMARY KEY,     -- 'medical.history.questions'
    label       VARCHAR NOT NULL,
    cefr        VARCHAR,                 -- 'A2', 'B1', 'B2', ...
    domain      VARCHAR,                 -- 'medical', 'academic', 'daily', ...
    parent_id   VARCHAR REFERENCES competencies(id)
);

-- Bayesian skill confidence per competency. mu/sigma name the Beta
-- posterior internally; the API surfaces `confidence = round(mu * 100)`
-- and `variance = round(sigma * 100)` as 0..100 ints for the UI.
CREATE TABLE IF NOT EXISTS mastery (
    competency_id   VARCHAR PRIMARY KEY REFERENCES competencies(id),
    mu              FLOAT NOT NULL DEFAULT 0.5,   -- posterior mean, 0..1
    sigma           FLOAT NOT NULL DEFAULT 0.25,  -- posterior std, 0..0.5
    last_evidence   TIMESTAMP,
    evidence_count  INTEGER NOT NULL DEFAULT 0
);

-- Append-only log of every quality observation feeding the mastery model.
-- This is the source of truth; mastery rows are derivable by replaying
-- the ledger. Source = 'conversation' | 'drill' | 'reader' | 'capture'
-- | 'agent_pre' | 'agent_post' (whoever inserted the row).
CREATE TABLE IF NOT EXISTS evidence (
    id              VARCHAR PRIMARY KEY,
    competency_id   VARCHAR REFERENCES competencies(id),
    occurred_at     TIMESTAMP DEFAULT now(),
    surface_form    VARCHAR,                       -- what the learner produced
    quality         FLOAT NOT NULL,                -- 0..1 outcome
    source          VARCHAR NOT NULL,
    notes           TEXT
);
CREATE INDEX IF NOT EXISTS idx_evidence_competency
    ON evidence(competency_id, occurred_at);

-- 2026-04-30: Atlas planner (PIVOT_ROADMAP §6.1, deferred from A.7 first
-- slice). The curriculum planner (Phase B.2) writes one row per generated
-- plan; ``superseded_by`` lets us keep the full history as the goal /
-- mastery shifts without ever rewriting old rows.
CREATE TABLE IF NOT EXISTS atlas_plans (
    id            VARCHAR PRIMARY KEY,
    goal_id       VARCHAR REFERENCES goals(id),
    horizon       VARCHAR,                       -- '30d' | '60d' | '90d'
    plan          JSON NOT NULL,                 -- generated curriculum
    created_at    TIMESTAMP DEFAULT now(),
    superseded_by VARCHAR                        -- self-reference, no FK to allow forward writes
);
CREATE INDEX IF NOT EXISTS idx_atlas_plans_goal
    ON atlas_plans(goal_id, created_at);

-- 2026-04-30: review scheduler (PIVOT_ROADMAP §A.8). Phase A surfaces a
-- posterior-driven `next_review_at` so the Atrium can sort competencies
-- by what's due next without recomputing from mu/sigma every read.
ALTER TABLE mastery ADD COLUMN IF NOT EXISTS next_review_at TIMESTAMP;
CREATE INDEX IF NOT EXISTS idx_mastery_next_review ON mastery(next_review_at);
