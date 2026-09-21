-- ZenkAI DuckDB schema (source of truth).
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

CREATE INDEX IF NOT EXISTS idx_paragraphs_work_chapter
    ON paragraphs(work_id, chapter, position);
CREATE INDEX IF NOT EXISTS idx_occurrences_paragraph
    ON word_occurrences(paragraph_id, position);
CREATE INDEX IF NOT EXISTS idx_occurrences_word
    ON word_occurrences(word_id);

-- Migrations (append below, never edit above). Each ALTER is idempotent.

-- 2026-04: per-word character offsets in paragraphs.text. These make the
-- reader render with correct punctuation (paragraphs.text is the single
-- source of truth) and enable O(N) offset-based slicing without re-running
-- spaCy at serve time.
ALTER TABLE word_occurrences ADD COLUMN IF NOT EXISTS char_start INTEGER;
ALTER TABLE word_occurrences ADD COLUMN IF NOT EXISTS char_end   INTEGER;
CREATE INDEX IF NOT EXISTS idx_occurrences_paragraph_start
    ON word_occurrences(paragraph_id, char_start);

-- 2026-04: vocabulary harvester. External SRS owns scheduling (decision
-- 0001); vocab_queue is the authoritative learner state. The word_states
-- SRS table was dropped from the schema once its last reader moved to
-- vocab_queue (existing databases keep a stale copy, harmlessly).
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
