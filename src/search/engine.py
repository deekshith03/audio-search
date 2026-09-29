"""
Hybrid search over indexed transcripts (docs/PHASE_3_PLAN.md §4).

    query ─┬─ BM25 (pg_search) ─────┐
           ├─ trigram (pg_trgm) ────┤ keyword: mode "lexical"
           └─ dense (pgvector HNSW) ┤ semantic: mode "dense"      all three: mode "hybrid"
                                    ▼
            fusion (weighted RRF | convex) → optional cross-encoder rerank of the top
            `rerank_depth` → localize each to 1-3 sentences (tightened to the matched words for
            short keyword queries) → collapse overlapping / adjacent spans → top_k

The reranker runs in hybrid mode only: lexical and dense are the retrieval ablations the eval gate
compares against, and a cross-encoder would add a semantic signal to the keyword-only baseline.

    uv run python -m src.search.engine "why were arrays added to postgres" [--mode dense] [--top-k 5]

Results follow the eval harness contract: file_id, speaker (human name when labeled, else the
SPEAKER_xx label), start_seconds, end_seconds, plus text, highlight (<mark> around keyword
matches) and score. Speaker names are joined at query time, so renaming never needs re-indexing.
"""

import argparse
import html
import json
import re
import sys
import time
from dataclasses import dataclass, field, fields
from typing import Any, Dict, List, Optional, Sequence, Tuple

from psycopg2 import sql

from src.db.connection import connect
from src.search.chunkers import CHUNK_CONFIGS, CONTEXT_CONFIGS
from src.search.embedders import MODELS, Embedder, variant_key
from src.search.fusion import FUSIONS, fuse
from src.search.indexer import vector_literal
from src.search.localize import ScoredSentence, blend, collapse_spans, select_span, tighten_to_keywords
from src.search.rerankers import RERANKERS, Reranker

MODES = {"lexical": ("bm25", "trigram"), "dense": ("dense",), "hybrid": ("bm25", "trigram", "dense")}
DEFAULT_WORKSPACES = ("dataset",)
WORD_PATTERN = re.compile(r"\w+")


@dataclass(frozen=True)
class SearchConfig:
    """Everything the dev grid varies, plus fixed knobs with defaults. The defaults are a
    placeholder until the grid (step 6) picks the configuration."""

    chunker: str = "A-30s"
    model: str = "bge-small"
    context: bool = True
    fusion: str = "rrf"
    bm25_weight: float = 1.0
    trigram_weight: float = 1.0
    dense_weight: float = 1.0
    reranker: Optional[str] = None
    candidates: int = 50
    rerank_depth: int = 20
    rrf_k: int = 60
    trigram_threshold: float = 0.4
    trigram_max_query_words: int = 3
    hnsw_ef_search: int = 200
    span_max_sentences: int = 3
    span_max_seconds: float = 20.0
    span_min_seconds: float = 4.0
    span_extend_ratio: float = 0.8
    keyword_span_padding_seconds: float = 2.0
    dedupe_gap_seconds: float = 2.0

    def __post_init__(self):
        if self.chunker not in CHUNK_CONFIGS:
            raise ValueError(f"unknown chunker {self.chunker!r}; choose from {CHUNK_CONFIGS}")
        if self.model not in MODELS:
            raise ValueError(f"unknown model {self.model!r}; choose from {sorted(MODELS)}")
        if self.fusion not in FUSIONS:
            raise ValueError(f"unknown fusion {self.fusion!r}; choose from {sorted(FUSIONS)}")
        if self.reranker is not None and self.reranker not in RERANKERS:
            raise ValueError(f"unknown reranker {self.reranker!r}; choose from {sorted(RERANKERS)} or None")

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "SearchConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown search config key(s): {unknown}")
        return cls(**values)

    @property
    def uses_context(self) -> bool:
        return self.context and self.chunker in CONTEXT_CONFIGS

    @property
    def embedding_variant(self) -> str:
        return variant_key(self.model, self.uses_context)

    @property
    def weights(self) -> Dict[str, float]:
        return {"bm25": self.bm25_weight, "trigram": self.trigram_weight, "dense": self.dense_weight}


@dataclass
class SearchResponse:
    results: List[Dict[str, Any]]
    timings_ms: Dict[str, float] = field(default_factory=dict)
    fused_chunk_ids: List[int] = field(default_factory=list)


class SearchEngine:
    """Holds one database connection and lazily loaded models; safe to reuse across queries."""

    def __init__(self, conn=None, embedders: Optional[Dict[str, Any]] = None, rerankers: Optional[Dict[str, Any]] = None):
        self._conn = conn
        self._embedders: Dict[str, Any] = dict(embedders or {})
        self._rerankers: Dict[str, Any] = dict(rerankers or {})

    @property
    def conn(self):
        if self._conn is None or self._conn.closed:
            self._conn = connect()
        return self._conn

    def embedder(self, key: str):
        if key not in self._embedders:
            self._embedders[key] = Embedder(key)
        return self._embedders[key]

    def reranker(self, key: str):
        if key not in self._rerankers:
            self._rerankers[key] = Reranker(key)
        return self._rerankers[key]

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int = 5,
        config: Optional[SearchConfig] = None,
        workspaces: Sequence[str] = DEFAULT_WORKSPACES,
    ) -> SearchResponse:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {sorted(MODES)}")
        config = config or SearchConfig()
        query = query.strip()
        response = SearchResponse(results=[])
        if not WORD_PATTERN.search(query):
            return response

        timings = response.timings_ms
        clock = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal clock
            now = time.perf_counter()
            timings[name] = round((now - clock) * 1000, 2)
            clock = now

        retrievers = [r for r in MODES[mode] if config.weights[r] > 0]
        query_vector = None
        if "dense" in retrievers:
            query_vector = self.embedder(config.model).encode_query(query)
            lap("embed_query")

        try:
            with self.conn.cursor() as cur:
                lists: Dict[str, List[Tuple[int, float]]] = {}
                if "bm25" in retrievers:
                    lists["bm25"] = self._bm25(cur, query, config, workspaces)
                if "trigram" in retrievers and len(WORD_PATTERN.findall(query)) <= config.trigram_max_query_words:
                    lists["trigram"] = self._trigram(cur, query, config, workspaces)
                if query_vector is not None:
                    lists["dense"] = self._dense(cur, query_vector, config, workspaces)
                lap("retrieve")

                fused = fuse(config.fusion, lists, config.weights, config.rrf_k)
                response.fused_chunk_ids = [chunk_id for chunk_id, _ in fused]
                shortlist = fused[: config.rerank_depth]
                chunks = self._chunk_rows(cur, [chunk_id for chunk_id, _ in shortlist])
                ranked = [(chunks[chunk_id], score) for chunk_id, score in shortlist if chunk_id in chunks]
                lap("fuse")

                if config.reranker and mode == "hybrid" and ranked:
                    scores = self.reranker(config.reranker).score(query, [self._rerank_input(c, config) for c, _ in ranked])
                    ranked = sorted(zip([c for c, _ in ranked], scores), key=lambda cs: -cs[1])
                    lap("rerank")

                spans = self._localize(cur, query, query_vector, ranked, mode, config)
                results = self._collapse(spans, config)[:top_k]
                self._highlight(cur, query, results)
                lap("localize")
            self.conn.rollback()
        except Exception:
            self.conn.rollback()
            raise

        for rank, r in enumerate(results, start=1):
            r["rank"] = rank
            r.pop("file_key")
        response.results = results
        timings["total"] = round(sum(timings.values()), 2)
        return response

    def _bm25(self, cur, query: str, config: SearchConfig, workspaces: Sequence[str]) -> List[Tuple[int, float]]:
        cur.execute(
            "SELECT c.id, pdb.score(c.id) FROM chunks c"
            " WHERE c.text ||| %(q)s AND c.chunker = %(chunker)s"
            " AND c.file_pk IN (SELECT id FROM files WHERE workspace = ANY(%(ws)s))"
            " ORDER BY pdb.score(c.id) DESC, c.id LIMIT %(n)s",
            {"q": query, "chunker": config.chunker, "ws": list(workspaces), "n": config.candidates},
        )
        return [(r[0], float(r[1])) for r in cur.fetchall()]

    def _trigram(self, cur, query: str, config: SearchConfig, workspaces: Sequence[str]) -> List[Tuple[int, float]]:
        cur.execute("SELECT set_config('pg_trgm.word_similarity_threshold', %s, true)", (str(config.trigram_threshold),))
        cur.execute(
            "SELECT c.id, word_similarity(%(q)s, c.text) AS sim FROM chunks c"
            " WHERE %(q)s <%% c.text AND c.chunker = %(chunker)s"
            " AND c.file_pk IN (SELECT id FROM files WHERE workspace = ANY(%(ws)s))"
            " ORDER BY sim DESC, c.id LIMIT %(n)s",
            {"q": query, "chunker": config.chunker, "ws": list(workspaces), "n": config.candidates},
        )
        return [(r[0], float(r[1])) for r in cur.fetchall()]

    def _dense(self, cur, vector: Sequence[float], config: SearchConfig, workspaces: Sequence[str]) -> List[Tuple[int, float]]:
        dims = MODELS[config.model].dimensions
        cur.execute("SELECT set_config('hnsw.ef_search', %s, true)", (str(config.hnsw_ef_search),))
        cur.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
        distance = sql.SQL("e.embedding::vector({d}) <=> %(v)s::vector({d})").format(d=sql.Literal(dims))
        cur.execute(
            sql.SQL(
                "SELECT e.chunk_id, 1 - ({distance}) AS sim FROM chunk_embeddings e JOIN chunks c ON c.id = e.chunk_id"
                " WHERE e.model = %(variant)s AND c.chunker = %(chunker)s"
                " AND c.file_pk IN (SELECT id FROM files WHERE workspace = ANY(%(ws)s))"
                " ORDER BY {distance} LIMIT %(n)s"
            ).format(distance=distance),
            {"v": vector_literal(vector), "variant": config.embedding_variant, "chunker": config.chunker,
             "ws": list(workspaces), "n": config.candidates},
        )
        return sorted(((r[0], float(r[1])) for r in cur.fetchall()), key=lambda kv: (-kv[1], kv[0]))

    def _chunk_rows(self, cur, chunk_ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
        if not chunk_ids:
            return {}
        cur.execute(
            "SELECT c.id, c.file_pk, f.workspace, f.file_id, c.speaker_label, sp.display_name,"
            " c.start_s, c.end_s, c.text, c.sentence_ids, c.context_text"
            " FROM chunks c JOIN files f ON f.id = c.file_pk"
            " LEFT JOIN speakers sp ON sp.file_pk = c.file_pk AND sp.speaker_label = c.speaker_label"
            " WHERE c.id = ANY(%s)",
            (list(chunk_ids),),
        )
        keys = ("id", "file_pk", "workspace", "file_id", "speaker_label", "display_name", "start_s", "end_s", "text",
                "sentence_ids", "context_text")
        return {r[0]: dict(zip(keys, r)) for r in cur.fetchall()}

    def _localize(self, cur, query, query_vector, ranked, mode: str, config: SearchConfig) -> List[Dict[str, Any]]:
        if not ranked:
            return []
        sentence_ids = sorted({sid for chunk, _ in ranked for sid in chunk["sentence_ids"]})
        use_dense = query_vector is not None
        use_keyword = mode != "dense"
        short_query = len(WORD_PATTERN.findall(query)) <= config.trigram_max_query_words
        dims = MODELS[config.model].dimensions
        similarity = (
            sql.SQL("1 - (se.embedding::vector({d}) <=> %(v)s::vector({d}))").format(d=sql.Literal(dims))
            if use_dense else sql.SQL("NULL::float")
        )
        cur.execute(
            sql.SQL(
                "WITH turns AS (SELECT DISTINCT file_pk, turn_id FROM sentences WHERE id = ANY(%(ids)s))"
                " SELECT s.id, s.file_pk, s.turn_id, s.start_s, s.end_s, s.text, {similarity},"
                " ts_rank_cd(to_tsvector('english', s.text), plainto_tsquery('english', %(q)s)),"
                " CASE WHEN %(short)s THEN word_similarity(%(q)s, s.text) END, s.words"
                " FROM sentences s JOIN turns USING (file_pk, turn_id)"
                " LEFT JOIN sentence_embeddings se ON se.sentence_id = s.id AND se.model = %(model)s"
                " ORDER BY s.file_pk, s.turn_id, s.start_s"
            ).format(similarity=similarity),
            {"ids": sentence_ids, "q": query, "short": short_query, "model": config.model,
             "v": vector_literal(query_vector) if use_dense else None},
        )
        turn_sentences: Dict[Tuple[int, int], List[tuple]] = {}
        for row in cur.fetchall():
            turn_sentences.setdefault((row[1], row[2]), []).append(row)

        position: Dict[int, Tuple[int, int]] = {}
        for key, rows in turn_sentences.items():
            for row in rows:
                position[row[0]] = key

        results = []
        for chunk, score in ranked:
            key = position.get(chunk["sentence_ids"][0]) if chunk["sentence_ids"] else None
            if key is None:
                continue
            rows = turn_sentences[key]
            signals: Dict[str, List[Optional[float]]] = {}
            if use_dense:
                signals["dense"] = [r[6] for r in rows]
            if use_keyword:
                signals["keyword"] = [max(float(r[7] or 0.0), float(r[8] or 0.0)) for r in rows]
            weights = {"dense": config.dense_weight, "keyword": config.bm25_weight}
            sentence_scores = blend(signals, weights)
            window = [ScoredSentence(r[0], r[3], r[4], r[5], s) for r, s in zip(rows, sentence_scores)]
            lo, hi = select_span(
                window, set(chunk["sentence_ids"]), config.span_max_sentences, config.span_max_seconds,
                config.span_min_seconds, config.span_extend_ratio,
            )
            span = window[lo:hi + 1]
            start_s, end_s, text = span[0].start_s, span[-1].end_s, " ".join(s.text for s in span)
            if use_keyword and short_query and config.keyword_span_padding_seconds > 0:
                words = [tuple(w) for r in rows[lo:hi + 1] for w in r[9]]
                tightened = tighten_to_keywords(words, query, config.keyword_span_padding_seconds)
                if tightened:
                    w_lo, w_hi = tightened
                    start_s, end_s = words[w_lo][1], words[w_hi][2]
                    text = " ".join(w[0] for w in words[w_lo:w_hi + 1])
            results.append({
                "file_key": (chunk["workspace"], chunk["file_id"]),
                "file_id": chunk["file_id"],
                "workspace": chunk["workspace"],
                "speaker": chunk["display_name"] or chunk["speaker_label"],
                "speaker_label": chunk["speaker_label"],
                "start_seconds": round(start_s, 3),
                "end_seconds": round(end_s, 3),
                "text": text,
                "score": round(float(score), 6),
                "chunk_id": chunk["id"],
                "chunk_start_seconds": round(chunk["start_s"], 3),
                "chunk_end_seconds": round(chunk["end_s"], 3),
            })
        return results

    @staticmethod
    def _rerank_input(chunk: Dict[str, Any], config: SearchConfig) -> str:
        if config.uses_context and chunk.get("context_text"):
            return f"{chunk['context_text']}\n\n{chunk['text']}"
        return chunk["text"]

    @staticmethod
    def _collapse(spans: List[Dict[str, Any]], config: SearchConfig) -> List[Dict[str, Any]]:
        groups = collapse_spans(
            [(s["file_key"], s["speaker_label"], s["start_seconds"], s["end_seconds"]) for s in spans],
            config.dedupe_gap_seconds, config.span_max_seconds,
        )
        results = []
        for group in groups:
            kept = dict(spans[group[0]])
            if len(group) > 1:
                members = sorted((spans[i] for i in group), key=lambda s: s["start_seconds"])
                kept["start_seconds"] = members[0]["start_seconds"]
                kept["end_seconds"] = max(m["end_seconds"] for m in members)
                kept["text"] = " ".join(m["text"] for m in members)
            results.append(kept)
        return results

    def _highlight(self, cur, query: str, results: List[Dict[str, Any]]) -> None:
        if not results:
            return
        cur.execute(
            "SELECT ts_headline('english', t, plainto_tsquery('english', %s),"
            " 'StartSel=<mark>, StopSel=</mark>, HighlightAll=true') FROM unnest(%s::text[]) WITH ORDINALITY AS u(t, n) ORDER BY n",
            (query, [html.escape(r["text"], quote=False) for r in results]),
        )
        for r, (highlighted,) in zip(results, cur.fetchall()):
            r["highlight"] = highlighted


_default_engine: Optional[SearchEngine] = None


def default_engine() -> SearchEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = SearchEngine()
    return _default_engine


def search(
    query: str,
    mode: str = "hybrid",
    top_k: int = 5,
    config: Optional[SearchConfig] = None,
    workspaces: Sequence[str] = DEFAULT_WORKSPACES,
) -> List[Dict[str, Any]]:
    """Eval-harness entry point: the top_k localized results for `query`."""
    return default_engine().search(query, mode, top_k, config, workspaces).results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Search indexed transcripts.")
    parser.add_argument("query")
    parser.add_argument("--mode", default="hybrid", choices=sorted(MODES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--workspace", action="append", dest="workspaces", help="Repeatable; default dataset.")
    parser.add_argument("--config", default="{}", help='JSON overrides, e.g. \'{"chunker": "B", "reranker": "bge-reranker"}\'.')
    args = parser.parse_args(argv)

    config = SearchConfig.from_dict(json.loads(args.config))
    response = default_engine().search(args.query, args.mode, args.top_k, config, args.workspaces or DEFAULT_WORKSPACES)
    for r in response.results:
        print(f"{r['rank']}. {r['file_id']}  {r['speaker']}  [{r['start_seconds']:.1f}-{r['end_seconds']:.1f}s]  score {r['score']:.4f}")
        print(f"   {r['highlight']}")
    print(f"timings (ms): {response.timings_ms}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
