-- Localization index: one vector per sentence per model (sentence text only, no context).
-- Independent of the chunk configs, so dropping a losing chunker never breaks localization.
-- Lookups are by sentence id (a handful per result), so no ANN index is needed.
CREATE TABLE sentence_embeddings (
    sentence_id BIGINT NOT NULL REFERENCES sentences(id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    embedding   vector NOT NULL,
    PRIMARY KEY (sentence_id, model)
);
