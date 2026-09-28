-- pg_search: BM25 keyword ranking. vector: dense embeddings (HNSW). pg_trgm: fuzzy matching
-- for ASR misspellings ("Hyperland" for "Hyprland"). ParadeDB preinstalls the first two in the
-- default database; IF NOT EXISTS keeps this safe on any Postgres that ships the extensions.
CREATE EXTENSION IF NOT EXISTS pg_search;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
