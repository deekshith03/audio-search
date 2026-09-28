# Phase 1 Summary: Dataset Curation & Evaluation Architecture

This document provides complete context on the work executed, architectural decisions made, and artifacts created during **Phase 1 (Golden Dataset Preparation & Evaluation Framework)** for the **Conversational Audio Hybrid Search** system.

---

## 1. Problem Framing & Objectives

The goal is to design, implement, and evaluate a local, production-grade hybrid search engine (Lexical + Semantic) across conversational audio recordings.

### Key Retrieval Requirements:
1. **Search Query Types:** Support exact keyword / full-text queries (Type A) and abstract semantic / conceptual paraphrases (Type B).
2. **Result Contract:** Every result must pinpoint:
   - Containing `file_id`
   - Spoken conversational turn interval `[start_seconds, end_seconds]` (exact jump-to audio offset)
   - Attributed `speaker`
   - Highlighted spoken text with `<mark>...</mark>` tags
3. **Evaluation Standard:** Automated test suite measuring **Recall@k** ($k=1, 3, 5$), **MRR**, and **Near-Miss Distractor Rejection** across a cross-file benchmark.

---

## 2. Standardized Golden Dataset (7 Audio Files)

Located in `dataset/audio/`. All files are standardized to **16,000 Hz, 1-channel Mono, 16-bit PCM WAV** with durations between **08:26 and 09:44 minutes**:

| File ID | Domain / Topic | Source / Show | Unique Two-Speaker Pair | Acoustic Environment | Duration |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`audio_01_lex_dhh_omarchy.wav`** | Software Systems & OS Engineering | *Lex Fridman #501* | Lex Fridman + DHH | Studio Clean (High SNR) | `09:44` (584.0s) |
| **`audio_02_postgres_fm_pgvector.wav`** | Database Internals & Vector Search | *Postgres FM (Ep. 74)* | Michael Christofides + Jonathan S. Katz | Broadcast Internet Call | `09:19` (559.0s) |
| **`audio_03_brain_science_psychology.wav`** | Psychology & Cognitive Neuroscience | *Changelog: Brain Science (Ep. 1)* | Adam Stacoviak + Dr. Mireille B. Reece | Studio Remote Call | `09:00` (540.0s) |
| **`audio_04_macroeconomics_semiconductor.wav`** | Macroeconomics & Hardware CapEx | *Dwarkesh Patel Podcast* | Dwarkesh Patel + Dylan Patel | High-SNR Studio | `09:01` (541.6s) |
| **`audio_05_sports_tactics_debate.wav`** | Combat Sports Tactics & Strategy | *Lex Fridman #500* | Lex Fridman + Khabib Nurmagomedov | Studio Broadcast | `09:00` (540.0s) |
| **`audio_06_constitutional_jurisprudence.wav`** | Constitutional Law & Jurisprudence | *Conversations with Tyler (Ep. 262)* | Tyler Cowen + Cass Sunstein | Faculty Office Studio | `08:26` (506.0s) |
| **`audio_07_investigative_history_cafe.wav`** | Investigative Military History | *Lex Fridman #499* | Lex Fridman + Gary Gallagher | **Authentic Cafe Chatter (12 dB SNR)** | `08:54` (534.0s) |

> **Provenance:** `dataset/metadata/sources.json` records each file's title, speakers, transcript source, output format and SHA-256; `python -m scripts.build_dataset --verify` checks the committed audio against it. The source audio URLs, clip offsets and audio licenses were not recorded during curation and are `null` there, so `--build` cannot yet recreate the clips from scratch.

> **Acoustic Robustness (`audio_07`):** Mixed with real environmental cafe chatter (background voices, coffee cups, dishes, room walla) at a calibrated **12.0 dB SNR** from the DEMAND acoustic dataset (CC BY 4.0 license, preserved in `dataset/noise_profiles/authentic_cafe_chatter_16k.wav`).

---

## 3. Ground-Truth Transcripts (`dataset/ground_truth/`)

Seven canonical JSON files (`audio_01_...json` to `audio_07_...json`) serve as the **Oracle ground truth**:
- **164 total conversational turns.**
- **100% Strict Turn Continuity:** Adjacent turns have exact temporal equality ($t1_{\text{end}} == t2_{\text{start}}$) with **0.0s gaps and 0.0s overlaps**.
- **Finite, Positive Durations:** All turns satisfy $t_{\text{start}} < t_{\text{end}}$ and are strictly bounded by physical audio duration.
- **Role & Speaker Metadata:** Explicit mapping of speaker identity to conversational role.
- **Timing provenance:** text is from each show's official transcript (link in `provenance`). Lex Fridman transcripts publish per-paragraph timestamps; the other four shows do not publish per-turn times, so those were assigned during curation without using Whisper or pyannote output, keeping the references independent of the Phase 2 pipeline.

---

## 4. Benchmark Evaluation Queries (`dataset/qrels/benchmark_queries.json`)

To prevent the evaluation from degrading into simple single-file topic classification, the benchmark contains **18 cross-file queries** structured across three distinct evaluation categories:

### A. Single-File Queries (6 queries — Needle-in-a-Haystack Isolation)
The answer exists in strictly one file. The remaining 6 files act as background noise and distractors:
- `SF-01`: Wayland/Hyprland tiling compositor (`audio_01`, Turn 5)
- `SF-02`: PostgreSQL 8KB index page limit (`audio_02`, Turn 12)
- `SF-03`: The Little Mermaid dinglehopper metaphor for functional struggle (`audio_03`, Turn 19)
- `SF-04`: 55,000 N3, 6K N5, and 170K DRAM wafers for 1GW compute (`audio_04`, Turn 11)
- `SF-05`: Joe Louis vs. Max Schmeling historic boxing match with Roosevelt/Hitler (`audio_05`, Turn 32)
- `SF-06`: Kantian liberalism requiring people to be treated as ends not means (`audio_06`, Turn 20)

### B. Multi-File Queries (6 queries — Cross-Corpus Recall)
The concept spans 2 to 4 files. `Recall@k` measures whether the system recovers the target moments across multiple distinct recordings:
- `MF-01`: Rapid rise of AI reshaping infrastructure & workflows (`audio_01`, `audio_02`, `audio_04`)
- `MF-02`: Overcoming doubt, struggle, and personal adversity (`audio_01`, `audio_03`, `audio_05`, `audio_07`)
- `MF-03`: Economics and computational costs of running and scaling infrastructure (`audio_02`, `audio_04`)
- `MF-04`: Wartime political leaders backing military commanders or national champions (`audio_05`, `audio_07`)
- `MF-05`: American GDP growth driven by technological breakthroughs (`audio_01`, `audio_04`)
- `MF-06`: State legal boundaries and physical coercion impacting human bodies (`audio_03`, `audio_06`)

### C. Near-Miss Queries (6 queries — Semantic Precision & Distractor Rejection)
Superficially similar language appears in multiple files, but only one file is the true answer. Evaluates whether the retriever avoids deceptive distractors:
- `NM-01`: Engineer assuming new database system is broken due to approximate results (`audio_02` true target vs. `audio_01` AI code distractor)
- `NM-02`: Elementary school personal memory regarding learning disabilities (`audio_03` true target vs. `audio_04` language assimilation distractor)
- `NM-03`: Athlete pacing energy to exhaust opponent stamina (`audio_05` true target vs. `audio_03` psychological struggle distractor)
- `NM-04`: Database index scan vs. query without any index (`audio_02` true target vs. `audio_04` hardware compute index distractor)
- `NM-05`: Why General Grant was labeled a butcher (`audio_07` true target vs. `audio_05` MMA combat violence distractor)
- `NM-06`: Kantian dignity under state enforcement (`audio_06` true target vs. `audio_04` immigration distractor)

> **Verbatim Substring Integrity:** 100% of all 29 target moments across all 18 queries are verified continuous verbatim substrings of their referenced turns.

> **Moment spans (tightened in Phase 2 remediation):** each relevant moment's `start_seconds` / `end_seconds` now cover only its `matched_text`, not the whole turn (turns run up to 180 s while matched text is often under 10% of the turn). Spans come from force-aligning the ground-truth turn text to the audio with wav2vec2 (`scripts/tighten_qrels.py`), independent of pipeline outputs; the original turn bounds are kept as `turn_start_seconds` / `turn_end_seconds`. As a QA check, 27 of 29 spans agree with the pipeline transcript's placement of the same words to within 1.3 s; the other two (both MF-01) are QA-matcher misses caused by ASR errors, not span errors: "pgvector" transcribed as "PG vector" (audio_02) and "OpenAI and Anthropic" as "OpenAnthropic" (audio_04). Hard-negative spans remain whole turns.

---

## 5. Evaluation Harness & Tooling Architecture

The evaluation harness runs via **Promptfoo** and a standalone **Mathematical Scorer**:

```
                                ./evals/run_evals.sh
                                         │
        ┌────────────────────────────────┴────────────────────────────────┐
        ▼                                                                 ▼
[ STAGE 1: Integrity Check ]                               [ STAGE 2: Promptfoo Matrix ]
• uv run python evals/validate_dataset_integrity.py         • npx promptfoo eval
• Validates WAV headers (16kHz, mono, PCM)                 • 54 automated tests (18 queries x 3 modes)
• Validates transcript continuity & non-zero turns         • Pure Python assertions (evals/assertions.py)
• Validates qrel containment & verbatim text               • Results UI: `bash evals/run_evals.sh --view`
        │                                                                 │
        └────────────────────────────────┬────────────────────────────────┘
                                         ▼
                           [ STAGE 3: Mathematical Scorer ]
                           • uv run python evals/evaluate_recall.py
                           • Computes Micro Recall@1, 3, 5 (per moment)
                           • Computes Macro Recall@1, 3, 5 (per query)
                           • Computes Mean Reciprocal Rank (MRR)
                           • Computes Near-Miss Rejection Rate
                           • Enforces Blueprint Gate (--enforce-gate):
                             - Micro Recall@5 >= 85.0%
                             - Micro Recall@1 >= 60.0%
                             - Strict Ablation: Hybrid > Lexical AND Hybrid > Dense
```

### Module Responsibilities:
- `promptfooconfig.yaml` & `promptfooconfig.json`: Declarative Promptfoo configuration generated with zero external dependencies by `evals/generate_promptfoo_config.py` (verified 100% structurally identical).
- `evals/assertions.py`: Pure Python assertion module. Requires strictly positive temporal overlap (`overlap > 0`), temporal tolerance ($IoU \ge 0.3$ or $|start_{\text{pred}} - start_{\text{gt}}| \le 5.0\text{s}$ with $\ge 25\%$ coverage), speaker match, and rejects hard-negative distractors at Rank #1.
- `evals/metrics.py`: Reusable mathematical formulas for IoU, delta matching, and ranking metrics.
- `evals/search_provider.py`: Adapter connecting Promptfoo to the search engine. Implements fail-closed behavior (returns critical error if `search_engine.py` is missing, unless `--mock` is explicitly passed).
- `tests/test_eval_metrics.py`: Standalone unit tests (discovered by `uv run python -m unittest discover -v`) verifying temporal matching, IoU partial calculations (1/3), speaker attribution, JSON/YAML parity, and gate checks (7/7 tests passing).

---

## 6. Oracle Audit & Clearance History

The evaluation setup was subjected to independent audits by **`@three-eyed-raven`**:
1. **Initial Audit (Verdict: HOLD):** Identified issues including unpinned dependencies, `|| true` swallowing failures, coarse whole-turn qrels, duplicate `dataset/references/` vs `dataset/ground_truth/`, and a 1.0s gap in `audio_03`.
2. **Intermediate Remediations:** Consolidated canonical ground truth, fixed the 1.0s gap in `audio_03`, rewrote qrels to exact continuous verbatim substrings, converted inlined JavaScript assertions to pure Python, and hardened the test runner with `set -euo pipefail`.
3. **Final Clearance Audit (Verdict: CLEARED):**
   - Verified 7 audio files and 7 continuous transcripts (zero defects).
   - Verified 7/7 unit tests passing in `tests/test_eval_metrics.py`.
   - Verified fail-closed runner exit codes (`100` without search engine, `1` with `--mock --enforce-gate`).
   - Verified JSON/YAML parity and qrel synchronization.
   - **Phase 1 officially declared CLEARED.**

---

## 7. Next Phases Roadmap

Phase 2 was implemented with different choices than originally planned here (faster-whisper `large-v3-turbo` fp32 on CPU, no Silero VAD, no LUFS normalization); see `docs/PHASE_2_SPECIFICATION.md` for the as-built design and rationale.

```
[Phase 2: Transcription Pipeline]                 → docs/PHASE_2_SPECIFICATION.md
[Phase B: Streamlit upload + speaker labeling]
[Phase C: Docker single-command setup]
[Phase 3: PostgreSQL + pgvector hybrid search, RRF fusion, <mark> highlighting]
[Phase 4: ./evals/run_evals.sh --enforce-gate (Recall@5 >= 0.85, Recall@1 >= 0.60, hybrid > lexical and dense)]
```
