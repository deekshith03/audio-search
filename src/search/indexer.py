"""
Indexes canonical transcripts into the search database: files, speakers, sentences (with word
timings), 30 s chunks, their EmbeddingGemma vectors, and per-sentence vectors (the localization
index).

    uv run python -m src.search.indexer                          # golden set
    uv run python -m src.search.indexer --workspace data --file x.wav
    uv run python -m src.search.indexer --no-embed               # chunks only, no embeddings

A file is re-chunked only when its canonical transcript or the chunking settings change (or with
--force); rows are replaced in one transaction, so a crash never leaves a file half indexed.
Embeddings are added incrementally: only chunks and sentences without a vector get one. Speaker
names are re-synced from speaker_labels/ on every run.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from psycopg2 import sql
from psycopg2.extras import Json, execute_values

from src.db.connection import connect
from src.pipeline.common import GOLDEN, Workspace, file_base, sha256_file
from src.pipeline.labels import load_labels
from src.search import chunkers, sentences
from src.search.chunkers import CHUNKER, build_chunks
from src.search.embedders import Embedder
from src.search.sentences import split_transcript

INDEXER_VERSION = 3
CANONICAL_SUFFIX = "_canonical.json"


def settings_key() -> str:
    settings = {"indexer": INDEXER_VERSION, "sentences": sentences.SETTINGS, "chunkers": chunkers.SETTINGS}
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def canonical_paths(workspace: Workspace, file_ids: Optional[Sequence[str]] = None) -> List[str]:
    if file_ids:
        paths = [workspace.canonical_path(file_base(f)) for f in file_ids]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"canonical transcript(s) not found: {missing}")
        return paths
    if not os.path.isdir(workspace.output_dir):
        return []
    return sorted(
        os.path.join(workspace.output_dir, f) for f in os.listdir(workspace.output_dir) if f.endswith(CANONICAL_SUFFIX)
    )


def vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def hnsw_index_name(model_key: str) -> str:
    return "chunk_embeddings_hnsw_" + re.sub(r"[^a-z0-9]+", "_", model_key.lower())


class Indexer:
    def __init__(self, conn, workspace: Workspace, embedders: Sequence[Any] = ()):
        self.conn = conn
        self.workspace = workspace
        self.embedders = list(embedders)
        self.settings_key = settings_key()

    def index_all(self, paths: Sequence[str], force: bool = False, prune: bool = False) -> List[Dict[str, Any]]:
        stats = [self.index_file(p, force=force) for p in paths]
        if prune:
            self.prune({s["file_id"] for s in stats})
        self.ensure_hnsw_indexes()
        return stats

    def index_file(self, canonical_path: str, force: bool = False) -> Dict[str, Any]:
        with open(canonical_path, encoding="utf-8") as f:
            canonical = json.load(f)
        file_id = canonical["file_id"]
        digest = sha256_file(canonical_path)
        stats: Dict[str, Any] = {"file_id": file_id, "rechunked": False, "chunks": 0, "embedded": {}}

        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, sha256, pipeline_key FROM files WHERE workspace = %s AND file_id = %s",
                (self.workspace.root, file_id),
            )
            row = cur.fetchone()
            if force or row is None or row[1] != digest or row[2] != self.settings_key:
                started = time.perf_counter()
                file_pk = self._rechunk(cur, canonical, digest)
                stats["rechunked"] = True
                stats["chunk_seconds"] = round(time.perf_counter() - started, 3)
            else:
                file_pk = row[0]
            self._sync_speakers(cur, file_pk, file_id, canonical.get("speaker_labels", []))
            cur.execute("SELECT count(*) FROM chunks WHERE file_pk = %s", (file_pk,))
            stats["chunks"] = cur.fetchone()[0]
        self.conn.commit()

        for embedder in self.embedders:
            started = time.perf_counter()
            count = self._embed_missing(file_pk, embedder) + self._embed_missing_sentences(file_pk, embedder)
            self.conn.commit()
            stats["embedded"][embedder.model.key] = {"vectors": count, "seconds": round(time.perf_counter() - started, 3)}
        return stats

    def _rechunk(self, cur, canonical: Dict[str, Any], digest: str) -> int:
        file_id = canonical["file_id"]
        sents = split_transcript(canonical)
        chunks = build_chunks(canonical, sents)

        cur.execute("DELETE FROM files WHERE workspace = %s AND file_id = %s", (self.workspace.root, file_id))
        cur.execute(
            "INSERT INTO files (workspace, file_id, display_name, duration_s, sha256, pipeline_key)"
            " VALUES (%s, %s, %s, %s, %s, %s) RETURNING id",
            (self.workspace.root, file_id, file_base(file_id), canonical.get("audio_duration_seconds"), digest, self.settings_key),
        )
        file_pk = cur.fetchone()[0]
        turn_words = {t["turn_id"]: t["words"] for t in canonical["turns"]}

        def word_timings(s):
            return Json([[w["word"], w["start_seconds"], w["end_seconds"]] for w in turn_words[s.turn_id][s.word_start:s.word_end]])

        sentence_ids = [
            r[0]
            for r in execute_values(
                cur,
                "INSERT INTO sentences (file_pk, speaker_label, turn_id, start_s, end_s, text, words) VALUES %s RETURNING id",
                [(file_pk, s.speaker_label, s.turn_id, s.start_s, s.end_s, s.text, word_timings(s)) for s in sents],
                fetch=True,
                page_size=1000,
            )
        ]
        execute_values(
            cur,
            "INSERT INTO chunks (file_pk, chunker, speaker_label, start_s, end_s, text, sentence_ids) VALUES %s",
            [
                (file_pk, CHUNKER, c.speaker_label, c.start_s, c.end_s, c.text, [sentence_ids[i] for i in c.sentence_indexes])
                for c in chunks
            ],
            page_size=1000,
        )
        return file_pk

    def _sync_speakers(self, cur, file_pk: int, file_id: str, speaker_labels: Sequence[str]) -> None:
        labels_doc = load_labels(self.workspace, file_id)
        names = labels_doc["labels"] if labels_doc else {}
        execute_values(
            cur,
            "INSERT INTO speakers (file_pk, speaker_label, display_name) VALUES %s"
            " ON CONFLICT (file_pk, speaker_label) DO UPDATE SET display_name = EXCLUDED.display_name",
            [(file_pk, label, names.get(label)) for label in speaker_labels],
        )

    def _embed_missing(self, file_pk: int, embedder) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT c.id, c.text FROM chunks c WHERE c.file_pk = %s"
                " AND NOT EXISTS (SELECT 1 FROM chunk_embeddings e WHERE e.chunk_id = c.id AND e.model = %s)"
                " ORDER BY c.id",
                (file_pk, embedder.model.key),
            )
            rows = cur.fetchall()
            if not rows:
                return 0
            vectors = embedder.encode_documents([text for _, text in rows])
            execute_values(
                cur,
                "INSERT INTO chunk_embeddings (chunk_id, model, embedding) VALUES %s",
                [(chunk_id, embedder.model.key, vector_literal(v)) for (chunk_id, _), v in zip(rows, vectors)],
                page_size=500,
            )
        return len(rows)

    def _embed_missing_sentences(self, file_pk: int, embedder) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT s.id, s.text FROM sentences s WHERE s.file_pk = %s"
                " AND NOT EXISTS (SELECT 1 FROM sentence_embeddings e WHERE e.sentence_id = s.id AND e.model = %s)"
                " ORDER BY s.id",
                (file_pk, embedder.model.key),
            )
            rows = cur.fetchall()
            if not rows:
                return 0
            vectors = embedder.encode_documents([text for _, text in rows])
            execute_values(
                cur,
                "INSERT INTO sentence_embeddings (sentence_id, model, embedding) VALUES %s",
                [(sentence_id, embedder.model.key, vector_literal(v)) for (sentence_id, _), v in zip(rows, vectors)],
                page_size=500,
            )
        return len(rows)

    def ensure_hnsw_indexes(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT model, max(vector_dims(embedding)) FROM chunk_embeddings GROUP BY model")
            for model_key, dims in cur.fetchall():
                cur.execute(
                    sql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON chunk_embeddings"
                        " USING hnsw ((embedding::vector({})) vector_cosine_ops) WHERE model = {}"
                    ).format(sql.Identifier(hnsw_index_name(model_key)), sql.Literal(dims), sql.Literal(model_key))
                )
        self.conn.commit()

    def prune(self, keep_file_ids: set) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "DELETE FROM files WHERE workspace = %s AND NOT (file_id = ANY(%s))",
                (self.workspace.root, sorted(keep_file_ids)),
            )
            removed = cur.rowcount
        self.conn.commit()
        return removed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Index canonical transcripts into the search database.")
    parser.add_argument("--workspace", default=GOLDEN.root, help="Workspace root (dataset/ or data/).")
    parser.add_argument("--file", action="append", dest="files", help="file_id or audio path to index (repeatable).")
    parser.add_argument("--no-embed", action="store_true", help="Index chunks and sentences without embeddings.")
    parser.add_argument("--force", action="store_true", help="Re-chunk even if nothing changed.")
    args = parser.parse_args(argv)

    workspace = Workspace(args.workspace)
    paths = canonical_paths(workspace, args.files)
    conn = connect()
    try:
        indexer = Indexer(conn, workspace, [] if args.no_embed else [Embedder()])
        started = time.perf_counter()
        for stats in indexer.index_all(paths, force=args.force, prune=not args.files):
            embedded = ", ".join(f"{k} {v['vectors']} in {v['seconds']}s" for k, v in stats["embedded"].items())
            action = f"re-chunked in {stats['chunk_seconds']}s" if stats["rechunked"] else "unchanged"
            print(f"{stats['file_id']}: {stats['chunks']} chunks, {action}" + (f"; embedded {embedded}" if embedded else ""))
        print(f"indexed {len(paths)} file(s) in {time.perf_counter() - started:.1f}s")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
