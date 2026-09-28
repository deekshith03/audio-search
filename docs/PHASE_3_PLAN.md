# Phase 3 Plan: Hybrid Search

Decisions agreed before implementation, the evidence behind them, and the order of work.

---

## 1. Requirements

From the problem statement:

| Requirement | How it is met |
| :--- | :--- |
| Search for words actually said **and** semantically similar terms | BM25 keyword search + dense vector search, fused |
| Results highlight file, timestamp and speaker | pinpointed 1–3 sentence spans; speaker names joined from human labels; `<mark>` highlights |
| Embedding generation and indexing run locally | local sentence-transformers models; local PostgreSQL (ParadeDB + pgvector) |
| Automated recall@k tests on a labeled query set | existing promptfoo + recall harness, with a dev/test split |
| Consider scale and production metrics | index choices (BM25 index, HNSW), latency, and a metrics section in the write-up |

## 2. Constraints From Our Data

| Fact (measured) | Consequence |
| :--- | :--- |
| Target moments: median **10.6 s** (1–28 s) | results must be short spans |
| Match rule: IoU ≥ 0.3, or start within 5 s with ≥ 25% of the result overlapping | a correct 75 s chunk still fails (IoU 0.14); returning 3-turn chunks passed **1 of 29** targets |
| Turns: median 12.6 s, p90 67 s, max 180 s; some are one word | turns are too uneven to use as-is |
| Every result must show one speaker | indexed text never crosses a speaker change |
| ASR drops punctuation (87/110 segments in `audio_02`) and misspells names ("Hyperland" for "Hyprland") | sentence splits fall back to pauses; keyword search needs fuzzy matching |

## 3. What the Research Says

1. **Simple chunking is a strong default; complex chunking rarely pays off.** Semantic, late and LLM-contextual chunking "rarely achieve significant wins over simpler baselines"; token chunking is a strong default and light context enrichment is the cheapest improvement. [arXiv 2608.16586](https://arxiv.org/html/2608.16586v1). Chunk size and overlap alone move recall by up to 9%; evaluate on your own queries. [Chroma](https://www.trychroma.com/research/evaluating-chunking). Contextual retrieval keeps meaning better than late chunking but costs more compute; neither is a definitive winner. [arXiv 2504.19754](https://arxiv.org/pdf/2504.19754)
2. **Conversations need speaker-aware chunking.** Boundaries only at turns, long turns split at sentences keeping their speaker, whole-unit overlap, short tails merged back. [chonkie #658](https://github.com/feyninc/chonkie/issues/658). Short turns indexed alone become "content-free fragments"; add surrounding context. [OpenTranscribe #523](https://github.com/attevon-llc/OpenTranscribe/issues/523)
3. **Spoken content: retrieve broadly, then localize.** TREC Podcasts used overlapping 2-minute segments with BM25 + neural reranking, judged per segment. [TREC 2020](https://arxiv.org/pdf/2103.15953). Fixed windows are poor at exact boundaries; sentence-level span localization does much better, and spontaneous speech is hardest. [arXiv 2609.21844](https://arxiv.org/html/2609.21844)
4. **Context and reranking compound.** Top-20 retrieval failures fell 35% with contextual embeddings, 49% adding contextual BM25, 67% adding a reranker. [Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)
5. **Keyword ranking quality matters.** Native `ts_rank` ignores corpus statistics (no IDF). [ParadeDB](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual). BM25 held up best as corpora grew in a 2026 scaling study. [arXiv 2607.26497](https://arxiv.org/html/2607.26497v2)
6. **Industry practice (Gong).** Conversations are converted into sentences, each embedded as a 768-d vector with metadata (billions stored); the model reads dialog context (a question followed by "Unfortunately, no" implies a delay); exact keyword search remains a separate tool; question answering retrieves up to the top 100 calls, then reasons. Gong reports keyword tracking at ~50% recall/precision and up to 80% more occurrences with semantic trackers (vendor claim). [Pinecone case study](https://www.pinecone.io/customers/gong/), [Gong](https://www.gong.io/blog/introducing-generation-3-conversation-understanding), [Gong Help](https://help.gong.io/docs/understanding-ai-ask-anything)

### BM25 in one example

Rare words carry more evidence. From our transcripts (135 chunks of ~80 words):

| Word | Chunks containing it | BM25 weight (IDF) |
| :--- | :---: | :---: |
| `wayland` | 1 | 4.51 |
| `compositor` | 1 | 4.51 |
| `people` | 27 | 1.60 |
| `think` | 37 | 1.29 |

BM25 also saturates repeated terms and normalizes for chunk length; Postgres's built-in ranking does none of this. Its weakness: the query "Hyprland" matches nothing because the ASR wrote "Hyperland", which is why keyword search is paired with fuzzy matching and semantic search.

## 4. Design

```
canonical turns (single speaker, word-timed)
   │ chunker (one of four, chosen on dev)
   ▼
chunks ──► embedding input = [previous turn, other speaker] + chunk text
       └─► BM25 / trigram input = chunk text only       (hits stay on the right speaker)
   │
   │ 1. keyword: BM25 (ParadeDB pg_search) + trigram fuzzy (pg_trgm)
   │ 2. semantic: pgvector cosine (HNSW index)
   │ 3. fusion: weighted reciprocal rank fusion (k = 60)
   │ 4. rerank: cross-encoder on the top 20 (kept only if dev shows a gain)
   │ 5. localize: best 1–3 sentences inside each top chunk
   │ 6. de-duplicate overlapping hits
   ▼
file · speaker name · [start–end] · text with <mark>highlights</mark> · ▶ play from start
```

### Chunkers compared

| | Chunker | Unit searched | Rationale |
| :--- | :--- | :--- | :--- |
| A | Single-speaker windows (~30 s, sentence-bounded, 1-sentence overlap) + previous-turn context | window | research findings 2–4 |
| B | Sentence units + previous-turn context (Gong-style) | sentence | finding 6 |
| C | Fixed 512-token windows | window | literature default (finding 1) |
| D | One chunk per speaker turn | turn | simplest conversation-aware baseline |

All four return pinpointed sentence spans, so they are compared on retrieval quality alone.

### Embedding models compared (CPU, this machine, 135 chunks)

| Model | Size | Per chunk | Per query | Max input | Access |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `bge-small-en-v1.5` | 33M | 6 ms | 6.5 ms | 512 | open |
| `bge-base-en-v1.5` | 109M | 16 ms | 16 ms | 512 | open |
| `EmbeddingGemma-300m` | 308M | 25 ms | 25 ms | 2,048 | gated (terms accepted) |
| `Qwen3-Embedding-0.6B` | 596M | 537 ms | 221 ms | 32k | open |

Selection rule: the fastest model within ~2 points of the best dev recall@5.

### Storage

```
files     (file_id, workspace, display_name, duration, sha256, pipeline_key, indexed_at)
speakers  (file_id, speaker_label, display_name)           ← synced from speaker_labels/*.json
sentences (id, file_id, speaker_label, turn_id, start, end, text)
chunks    (id, file_id, chunker, speaker_label, start, end, text, context_text,
           embedding vector(d), sentence_ids)
indexes:  BM25 (pg_search) on chunks.text, GIN gin_trgm_ops on chunks.text,
          HNSW on chunks.embedding, (file_id, start)
```

Renaming a speaker updates only `speakers`; nothing is re-embedded or re-indexed.

## 5. Evaluation Protocol

| Set | File | Queries | Use |
| :--- | :--- | :---: | :--- |
| **dev** (40%) | `dataset/qrels/dev_queries.json` | 14 (4 single-file / 4 multi-file / 4 near-miss / 2 keyword) | all tuning decisions |
| **test** (60%) | `dataset/qrels/test_queries.json` (formerly `benchmark_queries.json`) | 21 (6 / 6 / 6 / 3) | run **once**, after the configuration is frozen |

- Dev queries were drafted by Claude from the ground truth and reviewed by the user. The validator enforces that dev and test reference disjoint turns (targets and hard negatives).
- **Keyword category (`short_keyword`)**: one- or two-word queries (e.g. `Neuralink`, `Roger Gracie`). Any listed occurrence counts as a hit and counts as one moment in micro recall. Two of them (`pgMustard` in dev, `Omakub` in test) target words the ASR misspelled ("PG Mustard", "Omacoup"), which exercises fuzzy matching.
- **No-answer queries were considered and dropped**: scoring them needs a confidence threshold, and with 2–3 such queries per split the metric would be too noisy to tune or gate on. Consequence: search always returns its top results, even for queries unrelated to the corpus (listed as a limitation).
- Every script defaults to `--split dev`; the test split runs only when asked for explicitly (`bash evals/run_evals.sh --split test --enforce-gate`).
- Tuned on dev: chunker (A–D), embedding model (4), context on/off, window length (15/30/45 s), RRF weights, reranker on/off.
- Reported: recall@1/3/5 (micro and macro), MRR, per-category recall, near-miss rejection, p50/p95 search latency, indexing time per file.
- Gate (from Phase 1, applied to test): hybrid recall@5 ≥ 0.85, recall@1 ≥ 0.60, hybrid strictly better than keyword-only and semantic-only.
- The match rule is not loosened.
- Speaker matching in every scorer resolves anonymous `SPEAKER_xx` labels through `dataset/speaker_labels/` (shared `evals/speakers.py`).

## 6. Order of Work

1. ✅ Dev queries drafted and reviewed (12 + 2 keyword), plus 3 keyword queries added to test.
2. ✅ Split qrels into dev/test files; `--split` in the scorer and eval runner; per-split validation and promptfoo configs; shared speaker resolution in the scorers.
3. Switch the `db` image to ParadeDB; schema and migrations.
4. Sentence splitter and chunkers A–D; indexer (canonical transcripts → database).
5. Keyword, semantic, fusion, reranker, localization, de-duplication; `src/search/engine.search(query, mode, top_k)` wired into `evals/search_provider.py`.
6. Dev-set tuning grid; select the configuration.
7. Streamlit search page; `indexing` job stage for uploads; label edits sync to `speakers`; model bootstrap for Docker.
8. One final test-set run; write-up (design, success criteria, results, limitations, coding-agent disclosure).

## 7. Known Risks

- **Small query sets.** 12 dev queries can still overfit; decisions favor simpler options when results are close.
- **ASR errors hurt keyword search.** Fuzzy matching helps partly; no hand-written term fixes that only help the benchmark.
- **ParadeDB is AGPL.** Fine for this project; a closed commercial product would need a license review or `pg_textsearch` (PostgreSQL license, needs Postgres 17+).
- **Qwen3 memory (≈ 3 GB) in Docker** alongside Whisper during processing; measured during tuning.
- **The task asks for 5–6 files; the golden set has 7.** Justified in the write-up.
