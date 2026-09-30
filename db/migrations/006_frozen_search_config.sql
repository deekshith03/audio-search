-- The dev-set tuning grid is over (README §3): only the frozen configuration
-- remains. Drop the losing chunk configs, embedding variants and their HNSW indexes, and the
-- context column that only the losing configs used.
DELETE FROM chunks WHERE chunker <> 'A-30s';
DELETE FROM chunk_embeddings WHERE model <> 'gemma';
DELETE FROM sentence_embeddings WHERE model <> 'gemma';

DO $$
DECLARE idx record;
BEGIN
    FOR idx IN SELECT indexname FROM pg_indexes
               WHERE tablename = 'chunk_embeddings' AND indexname LIKE 'chunk_embeddings_hnsw_%'
                 AND indexname <> 'chunk_embeddings_hnsw_gemma'
    LOOP
        EXECUTE format('DROP INDEX IF EXISTS %I', idx.indexname);
    END LOOP;
END $$;

ALTER TABLE chunks DROP COLUMN context_text;
ALTER TABLE chunks DROP CONSTRAINT chunks_chunker_check;
ALTER TABLE chunks ADD CONSTRAINT chunks_chunker_check CHECK (chunker = 'A-30s');

COMMENT ON COLUMN chunk_embeddings.model IS 'embedding model key (gemma)';
