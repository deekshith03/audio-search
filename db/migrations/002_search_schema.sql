-- Search index built from canonical transcripts. Speaker names live only in `speakers`, so a
-- rename never touches chunks, embeddings or indexes.

CREATE TABLE files (
    id           BIGSERIAL PRIMARY KEY,
    workspace    TEXT NOT NULL,                  -- 'dataset' (golden set) or 'data' (uploads)
    file_id      TEXT NOT NULL,                  -- '<name>.wav', as used by the eval harness
    display_name TEXT,
    duration_s   DOUBLE PRECISION,
    sha256       TEXT NOT NULL,
    pipeline_key TEXT NOT NULL,                  -- canonical transcript cache key; re-index when it changes
    indexed_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (workspace, file_id)
);

CREATE TABLE speakers (
    file_pk       BIGINT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    speaker_label TEXT NOT NULL,                 -- 'SPEAKER_00'
    display_name  TEXT,                          -- human name from speaker_labels/*.json
    PRIMARY KEY (file_pk, speaker_label)
);

CREATE TABLE sentences (
    id            BIGSERIAL PRIMARY KEY,
    file_pk       BIGINT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    speaker_label TEXT NOT NULL,
    turn_id       INTEGER NOT NULL,
    start_s       DOUBLE PRECISION NOT NULL,
    end_s         DOUBLE PRECISION NOT NULL,
    text          TEXT NOT NULL,
    CHECK (end_s >= start_s)
);
CREATE INDEX sentences_file_start_idx ON sentences (file_pk, start_s);

-- One row set per chunker so the dev grid can compare A-D side by side.
CREATE TABLE chunks (
    id            BIGSERIAL PRIMARY KEY,
    file_pk       BIGINT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    chunker       TEXT NOT NULL CHECK (chunker IN ('A', 'B', 'C', 'D')),
    speaker_label TEXT NOT NULL,                 -- a chunk never crosses a speaker change
    start_s       DOUBLE PRECISION NOT NULL,
    end_s         DOUBLE PRECISION NOT NULL,
    text          TEXT NOT NULL,                 -- keyword-searched text
    context_text  TEXT,                          -- previous turn (other speaker), embedding input only
    sentence_ids  BIGINT[] NOT NULL,             -- localization picks the best 1-3 of these
    CHECK (end_s >= start_s)
);
CREATE INDEX chunks_file_start_idx ON chunks (file_pk, start_s);
CREATE INDEX chunks_text_trgm_idx ON chunks USING gin (text gin_trgm_ops);

-- English stemming ("compositors" matches "compositor"); chunker and file_pk are indexed as
-- exact fields so filters are pushed into the BM25 scan instead of applied afterwards.
CREATE INDEX chunks_bm25_idx ON chunks
USING bm25 (id, (text::pdb.simple('stemmer=english')), (chunker::pdb.literal), file_pk)
WITH (key_field = 'id');

-- Models differ in dimension (384/768/1024), so the column is untyped. The indexer creates one
-- partial HNSW index per model, on the expression (embedding::vector(d)) WHERE model = '<name>'.
CREATE TABLE chunk_embeddings (
    chunk_id  BIGINT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
    model     TEXT NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (chunk_id, model)
);
