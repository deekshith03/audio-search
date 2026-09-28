-- Chunker values are configs such as 'A-30s' or 'C-512', not bare letters.
ALTER TABLE chunks DROP CONSTRAINT chunks_chunker_check;
ALTER TABLE chunks ADD CONSTRAINT chunks_chunker_check CHECK (chunker ~ '^[A-D](-[a-z0-9]+)?$');

COMMENT ON COLUMN files.sha256 IS 'SHA-256 of the canonical transcript that was indexed';
COMMENT ON COLUMN files.pipeline_key IS 'fingerprint of the sentence/chunk settings; a change triggers re-chunking';
COMMENT ON COLUMN chunk_embeddings.model IS 'model variant: <model> or <model>+ctx (previous turn prepended)';
