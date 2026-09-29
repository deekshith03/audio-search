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
7. **RRF is the safe default, not the best fusion.** A convex combination of normalized scores beat RRF in and out of domain, with one parameter tuned from few examples; tuned RRF generalizes poorly. [Bruch, ACM TOIS](https://arxiv.org/abs/2210.11934). RRF averaged 3.86% lower nDCG@10 than score-based fusion on six BEIR datasets. [OpenSearch](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/). Score-level BM25 + dense fusion added 9–17 Hit@1 points on conversation retrieval. [arXiv 2606.04194](https://arxiv.org/abs/2606.04194). LLM-weighted fusion ([DAT](https://arxiv.org/abs/2503.23013)) and learned fusion (needs 1,000+ labels) do not fit a local, 14-query setup.
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
   │ 1. keyword: BM25 (ParadeDB pg_search) + trigram fuzzy (pg_trgm, queries of ≤ 3 words)
   │ 2. semantic: pgvector cosine (HNSW index)
   │ 3. fusion: weighted RRF (k = 60) or min-max convex combination (chosen on dev)
   │ 4. rerank: cross-encoder on the top 20 (kept only if dev shows a gain)
   │ 5. localize: best 1–3 sentences inside each top chunk (sentence vectors + keyword rank)
   │ 6. de-duplicate overlapping hits
   ▼
file · speaker name · [start–end] · text with <mark>highlights</mark> · ▶ play from start
```

### Chunkers compared

| | Chunker | Unit searched | Rationale |
| :--- | :--- | :--- | :--- |
| A | Single-speaker windows (15 / 30 / 45 s, sentence-bounded, 1-sentence overlap) + previous-turn context | window | research findings 2–4 |
| B | Sentence units + previous-turn context (Gong-style) | sentence | finding 6 |
| C | Fixed 512-token windows (bge tokenizer, so identical for every model), 64-token overlap | window | literature default (finding 1) |
| D | One chunk per speaker turn | turn | simplest conversation-aware baseline |

All four return pinpointed sentence spans, so they are compared on retrieval quality alone. No chunk crosses a speaker turn. Sentences come from punctuation, then the longest pause (≥ 0.4 s) for pieces over 30 words / 15 s, then fragments under 4 words merge into a neighbour (golden set: 751 sentences, median 4.1 s). Context (A, B) is the last 60 words of the previous turn by the other speaker, embedding input only.

### Embedding models compared (CPU, this machine, 135 chunks)

| Model | Size | Per chunk | Per query | Max input | Access |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `bge-small-en-v1.5` | 33M | 6 ms | 6.5 ms | 512 | open |
| `bge-base-en-v1.5` | 109M | 16 ms | 16 ms | 512 | open |
| `EmbeddingGemma-300m` | 308M | 25 ms | 25 ms | 2,048 | gated (terms accepted) |
| `Qwen3-Embedding-0.6B` | 596M | 537 ms | 221 ms | 32k | open |

Selection rule: the fastest model within ~2 points of the best dev recall@5. **Qwen3 is deferred**: the first grid uses the other three; Qwen3 is tried only if those results suggest it could win.

### Fusion (chosen on dev)

| Method | Score | Parameters |
| :--- | :--- | :--- |
| Weighted RRF | `Σ wᵢ / (60 + rankᵢ)` | one weight per retriever |
| Convex combination | `Σ wᵢ · normᵢ(score)`, weights sum to 1; BM25 min-max normalized per query, trigram and cosine already in [0, 1] | one weight per retriever |

Three retrievers (BM25, trigram, dense) mean two free weights; with 14 dev queries this can overfit, so **RRF wins ties** (more robust out of domain). Both methods are reported side by side on dev and on the final test run.

### Reranker (optional stage, decided on dev)

A cross-encoder reads the query and each candidate together and re-orders the fused top 20. It targets look-alike results (the near-miss queries), and can only reorder the shortlist, so it helps recall@1 and MRR more than recall@5.

| Candidate | Size | License | Notes |
| :--- | :---: | :--- | :--- |
| `Qwen/Qwen3-Reranker-0.6B` (run as the `tomaarsen/Qwen3-Reranker-0.6B-seq-cls` CrossEncoder port) | 0.6B | Apache 2.0 | top open reranker in 2026 comparisons |
| `BAAI/bge-reranker-v2-m3` | 568M | open | common open default |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 22M | Apache 2.0 | fast option (replaces mxbai-rerank-base-v2: ~0.5B, not faster, needs its own package) |

Sources: [reranker comparison](https://futureagi.com/blog/best-rerankers-for-rag-2026/), [Qwen3 embedding/reranking](https://qwenlm.github.io/blog/qwen3-embedding/).

- **CPU latency (step 5b, measured).** Rerank time for the fused top 20 per dev query (14 queries, hybrid, this Mac's CPU), p50 / p95 ms, by chunk length:

  | Reranker | Load | B (~4 s chunks) | A-30s (~22 s) | D (turns, up to 180 s) |
  | :--- | :---: | :---: | :---: | :---: |
  | Qwen3-Reranker-0.6B (with its chat template) | 2.6 s | 1,165 / 1,253 | 2,241 / 2,493 | 6,181 / 9,724 |
  | bge-reranker-v2-m3 | 7.2 s | 288 / 335 | 841 / 958 | 2,855 / 4,466 |
  | ms-marco-MiniLM-L6-v2 | 0.7 s | 20 / 23 | 44 / 53 | 133 / 134 |

  Placeholder config (A-30s, bge-small+ctx, equal RRF), dev hybrid, micro R@1 / R@5 / macro MRR: none 0.474 / 0.526 / 0.657; MiniLM 0.421 / 0.579 / 0.595; bge 0.474 / 0.632 / 0.679; Qwen3 0.474 / 0.632 / 0.661. Qwen3 scored R@1 0.0 until its model-card prompt template was applied. The reranker runs in hybrid mode only, so the keyword-only and semantic-only ablations stay clean.
- **Keep rule:** keep the reranker only if it clearly improves dev recall@1 / MRR / near-miss rejection without hurting recall@5, and its latency is acceptable. Otherwise ship fusion alone and report the comparison.

### Engine changes from the Round 1 audit

A hit-by-hit audit of dev misses confirmed the scorer grades correctly (all qrel speakers resolvable, every decision consistent with the match rule, grid recall code identical to the harness). Of six misses for B · gemma, three were the right place with a span too wide (`pgMustard`: a 12.3 s sentence for a 1.6 s target), one a diarization error (Dylan's 4.5 s interjection merged into Dwarkesh's turn), two true retrieval misses. Four engine changes followed, applied to every config:

1. **Keyword span tightening**: for queries of ≤ 3 words, the span shrinks to the matched words ± 2 s (exact or fuzzy, 1–3 word runs so `pgMustard` matches "PG Mustard"), using word timings stored on `sentences` (migration 005).
2. **Reranker reads context**: with context on, the cross-encoder scores context + chunk, not the chunk alone.
3. **Adjacent-span collapse**: results of the same file and speaker within 2 s of a kept one are merged if the union fits 20 s, else dropped, so neighbouring sentences no longer fill the top 5.
4. **B-prev chunker**: one sentence per chunk, context = the other speaker's previous turn + the speaker's previous sentence.

### Storage

```
files            (id, workspace, file_id, display_name, duration_s, sha256, pipeline_key, indexed_at)
                 UNIQUE (workspace, file_id): an upload may reuse a golden file name
speakers         (file_pk, speaker_label, display_name)   ← synced from speaker_labels/*.json
sentences        (id, file_pk, speaker_label, turn_id, start_s, end_s, text)
chunks           (id, file_pk, chunker, speaker_label, start_s, end_s, text, context_text, sentence_ids)
chunk_embeddings (chunk_id, model, embedding vector)      ← untyped: models are 384/768/1024-d
sentence_embeddings (sentence_id, model, embedding)      ← localization index, independent of chunkers
indexes:  BM25 (pg_search, English stemmer) on chunks(text) with chunker and file_pk as filter fields,
          GIN gin_trgm_ops on chunks.text, (file_pk, start_s) on chunks and sentences,
          one partial HNSW per model on (embedding::vector(d)) WHERE model = '<name>', created by the indexer
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
- Tuned on dev: chunker (A–D), embedding model, context on/off, window length (15/30/45 s), fusion method (weighted RRF vs convex combination) and its weights, reranker on/off.
- **Grid, scored end to end.** Round 1: 30 candidates = 10 chunk variants (A-15s/A-30s/A-45s/B with and without context, C-512, D) × 3 models, each run through the whole pipeline (hybrid → default RRF → default reranker → localization) on dev. Keyword-only, semantic-only and fused recall@20 are recorded as diagnostics, not used to decide. Round 2: top 2 candidates × fusion method and weights. Round 3: winner × reranker off / 3 candidates. Close results go to the simpler option.
- **After freezing**, the losing chunkers, models, fusion method and reranker code and their database rows are deleted; only the winner ships, and the grid tables are kept as findings in the write-up.
- Reported: recall@1/3/5 (micro and macro), MRR, per-category recall, near-miss rejection, RRF vs convex combination (dev and test, report only; the frozen choice is the one gated), p50/p95 search latency, indexing time per file.
- Gate (from Phase 1, applied to test): hybrid recall@5 ≥ 0.85, recall@1 ≥ 0.60, hybrid strictly better than keyword-only and semantic-only.
- The match rule is not loosened.
- Speaker matching in every scorer resolves anonymous `SPEAKER_xx` labels through `dataset/speaker_labels/` (shared `evals/speakers.py`).

## 6. Order of Work

1. ✅ Dev queries drafted and reviewed (12 + 2 keyword), plus 3 keyword queries added to test.
2. ✅ Split qrels into dev/test files; `--split` in the scorer and eval runner; per-split validation and promptfoo configs; shared speaker resolution in the scorers.
3. ✅ `db` image is `paradedb/paradedb:0.25.10-pg17`; schema in `db/migrations/` applied by `src.db.migrate` (also at app start in Docker).
4. ✅ Sentence splitter, 6 chunk configs, 3 default embedders (+ Qwen3 registered), incremental indexer `src.search.indexer` (golden set: 1,877 chunks, 3,478 vectors per model, ~2 min for all three models).
5. ✅ `src/search/engine.py` (retrievers, `fusion.py`, `rerankers.py`, `localize.py`, dedupe, `<mark>` highlights) wired into `evals/search_provider.py`; every grid knob is a `SearchConfig` field, passed through provider config or `evaluate_recall.py --search-config`. Placeholder config (A-30s, bge-small+ctx, equal RRF, no reranker) on dev: hybrid micro R@5 52.6%, lexical 31.6%, dense 36.8%.
5b. ✅ Reranker candidates downloaded, CPU latency and a first dev comparison measured (§4 Reranker); real-model smoke tests skip when a model is not downloaded.
6. Dev-set tuning grid; select the configuration.
   - ✅ Model round (30 candidates, 3 models; `evals/results/grid_models.json`): EmbeddingGemma led or tied in every chunk family; context had no effect once the reranker ran.
   - ✅ Engine changes above; index rebuilt with gemma only.
   - ✅ Family round (11 families, gemma, bge reranker, equal RRF; `grid_families.json`): A-30s R@5 0.789 / R@1 0.579 (was 0.684 / 0.526 before the engine changes); C-512 0.684 / 0.632 (≡ D); A-45s below A-30s; B+ctx last.
   - ⏳ Joint round: A-15s, A-30s, A-30s+ctx, B, B-prev+ctx, C-512 × 10 fusion settings × 4 rerankers (240 configs); eligible if search p50 ≤ 1.6 s; the reranker must beat its no-reranker twin.
   - ✅ Joint round (240 configs): A-30s + gemma + RRF + bge-reranker led on the question-style dev set; Qwen3 no better at 2.5–7 s, MiniLM hurt R@1 (both removed). Span round: current span settings best.
   - ✅ Query sets rewritten as search queries (the brief asks for term search): exact / semantic / keyword styles plus 2 questions per category; dev expanded to 50 queries (63 moments) on unused turns; test (21) reworded with labels kept, plus 3 label corrections from an eval audit (pooling, speaker reachability, answers in test-only turns).
   - ✅ Finalists round on the new dev set (A-30s vs B-prev+ctx × 10 fusion × {none, bge} + span variants): the bge reranker now hurts (keyword R@5 1.00 → 0.78: it demotes exact matches for 1–2 word searches). Error check of the 12 misses: 6 query/label problems (corrected by transcript, not by score), 2 span-too-wide, 3 real retrieval misses, 1 diarization.
   - ✅ Cleanup: losing chunkers, embedding models, RRF, reranker, context variants, adjacent merge and the grid runner removed from the code; migration 006 drops their rows, HNSW indexes and `chunks.context_text`; unused models deleted from the local cache. Grid results stay in `evals/results/`. The cleaned engine reproduces the frozen dev results exactly (no per-query differences).
   - ✅ **Frozen: A-30s · EmbeddingGemma (no context) · convex fusion (BM25 1, trigram 1, dense 2) · no reranker · span 0.8 / 4 s · no adjacent merge.** Final dev (one run): hybrid micro R@5 0.806, R@1 0.677, MRR 0.878; lexical 0.565, dense 0.758 micro R@5; exact 0.929, semantic 0.625, question 0.556, keyword 1.00 R@5; search p50 65 ms / p95 107 ms.
7. Streamlit search page (with optional speaker and recording filters; `speaker_label` and `file_pk` are already on every chunk); `indexing` job stage for uploads; label edits sync to `speakers`; model bootstrap for Docker.
8. One final test-set run; write-up (design, success criteria, results, limitations, coding-agent disclosure).

## 7. Known Risks

- **Small query sets.** 12 dev queries can still overfit; decisions favor simpler options when results are close.
- **ASR errors hurt keyword search.** Fuzzy matching helps partly; no hand-written term fixes that only help the benchmark.
- **ParadeDB is AGPL.** Fine for this project; a closed commercial product would need a license review or `pg_textsearch` (PostgreSQL license, needs Postgres 17+).
- **Qwen3 memory (≈ 3 GB) in Docker** alongside Whisper during processing; measured during tuning.
- **The task asks for 5–6 files; the golden set has 7.** Justified in the write-up.
