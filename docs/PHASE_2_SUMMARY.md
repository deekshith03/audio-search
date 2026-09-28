# Phase 2 Summary: Transcription & Diarization Pipeline

Design and rationale: `docs/PHASE_2_SPECIFICATION.md`. This document records what was built, how it scored, and what is still open.

---

## 1. What the Pipeline Does

```
audio (any format, ≤10 min) ─► ingest ─► ASR ─► align ─► diarize ─► reconcile ─► canonical.json
                                 │        │       │         │           │
                              ffmpeg   faster-  wav2vec2  pyannote   word→speaker turns
                              16k mono whisper  (CPU)     k=2 (CPU)  (SPEAKER_00 / _01)
                                       fp32 CPU
```

- One backend everywhere (native and Docker): faster-whisper `large-v3-turbo` fp32, wav2vec2 `WAV2VEC2_ASR_BASE_960H`, pyannote `speaker-diarization-community-1`, all on CPU.
- Every stage caches its artifact under a content + config `cache_key`; `--file` runs a single upload, `--force` recomputes.
- Transcripts carry only anonymous speaker labels; names come from the (Phase B) labeling step.

---

## 2. Remediation Applied (audit → fix)

| Finding | Fix |
| :--- | :--- |
| MLX/Metal cannot run in Docker | ASR moved to faster-whisper fp32 CPU after a head-to-head benchmark (spec §6); `mlx`, `mlx-whisper` removed. |
| "Raw DER" was computed on refined output (raw RTTM was overwritten) | `{id}.raw.rttm` keeps pyannote's untouched output; DER reported for raw, refined and reconciled separately. |
| Caches keyed by filename only, so code/model changes silently reused stale artifacts | Per-stage `cache_key` = SHA-256(config + input hashes); `--force`. |
| Stages could only batch-process the whole folder | `--file` on every stage; `run_pipeline.sh` forwards arguments. |
| Alignment fallback read the wrong key (`end` vs `end_seconds`), so every untimed word started at the segment start | Pure `interpolate_untimed_words()` spreads untimed runs across the correct gap; unit-tested. |
| Alignment model mislabelled as `wav2vec2-large-960h` | Recorded from WhisperX's actual default, `WAV2VEC2_ASR_BASE_960H`. |
| Fallback telemetry could not distinguish digits/symbols | New `wildcard_aligned_words` (1–35 per file); true fallbacks remain 0. |
| Pipeline emitted `Speaker 1/2` names and a dead GT-based name-mapping function | Canonical output has `speaker_labels` only; dead code removed. |
| Smoothing mutated the list it was reading, so corrections could cascade | Decisions read the original assignments. |
| Word speaker accuracy used only the first token of hyphenated words, and difflib's autojunk heuristic discarded frequent words | All tokens counted; `autojunk=False`. |
| No upload ingestion | `ingest.py`: ffprobe validation (formats, 10 s–10 min, audio stream present), ffmpeg to 16 kHz mono PCM16, content-hash `file_id`. |
| qrel moments spanned whole turns (up to 180 s) | Narrowed to `matched_text` via reference-side forced alignment (Phase 1 summary §4). |
| Golden audio provenance not recorded | `dataset/metadata/sources.json` + `scripts/build_dataset.py --verify` (source URLs/offsets still to be filled in, §5). |
| Lockfile missing `whisper-normalizer`; loose `pyannote-audio>=3.1`, `whisperx>=3.1` bounds | Relocked; bounds set to the APIs actually used (`pyannote-audio>=4,<5`, `whisperx>=3.4.2,<3.5`). |
| Tests covered dead code and needed a real HF token | 105 tests, token tests mocked (§4). |

---

## 3. Quality Scorecard (golden set, fresh `--force` run)

Collar 0 ms unless noted; overlap scored.

| File | Std WER | S / D / I | Raw DER | Raw DER 250ms | Confusion | Refined DER | Reconciled DER | Word Spk Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `audio_01` Software OS | 3.56% | 27 / 19 / 10 | 11.36% | 11.10% | 1.35% | 11.23% | 3.47% | **99.93%** |
| `audio_02` Databases | 9.16% | 29 / 15 / 131 | 4.17% | 3.89% | 1.12% | 4.13% | 2.23% | **98.81%** |
| `audio_03` Psychology | 9.98% | 13 / 11 / 121 | 8.92% | 8.31% | 1.02% | 8.63% | 2.94% | **99.78%** |
| `audio_04` Hardware | 12.67% | 41 / 20 / 140 | 6.39% | 5.95% | 2.80% | 5.71% | 5.47% | **96.89%** |
| `audio_05` Sports | 1.08% | 6 / 3 / 6 | 18.62% | 17.75% | 5.64% | 16.82% | 8.93% | **95.20%** |
| `audio_06` Law | 3.28% | 5 / 7 / 32 | 3.96% | 3.41% | 0.82% | 3.90% | 2.65% | **99.24%** |
| `audio_07` Cafe, 12 dB SNR | 4.27% | 20 / 44 / 7 | 7.33% | 6.77% | 1.82% | 6.54% | 4.21% | **98.92%** |

| Mean | Std WER | Raw DER | Raw confusion | Raw missed | Raw FA | Refined DER | Reconciled DER | Word Spk Acc | Lexical coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Clean (01–06) | 6.62% | 8.90% | 2.13% | 6.51% | 0.27% | 8.40% | 4.28% | **98.31%** | 92.13% |
| Noisy (07) | 4.27% | 7.33% | 1.82% | 4.36% | 1.16% | 6.54% | 4.21% | **98.92%** | 89.88% |
| All 7 | 6.29% | 8.68% | 2.08% | 6.20% | 0.39% | 8.14% | 4.27% | **98.40%** | 91.81% |

### Reading these numbers

- **Word speaker accuracy (98.4%) is the headline speaker metric.** It is independent of reference timestamps and is exactly what search results expose: whether a hit shows the right speaker.
- **Most of raw DER is missed detection (6.2 of 8.7 points), not speaker confusion (2.1).** The reference turns are gap-free, so real pauses count as missed speech.
- **Reconciled DER is lower mainly because of that convention.** Reconciled turns bridge pauses up to 1.5 s, which matches the gap-free reference; it does not mean attribution improved.
- **Centroid refinement fired only on `audio_05`** (centroid similarity 0.47, 6 segments relabelled; raw 18.62% → refined 16.82% DER). Its thresholds were tuned on this same file, so treat that gain as optimistic.
- **WER on `audio_02`, `audio_03`, `audio_04` is dominated by insertions (121–140).** The published transcripts are edited (fillers and false starts removed), so many "insertions" are speech the reference omits. Substitutions, the errors that matter for search, are 5–41 per file.
- **`audio_05` is the hardest diarization file:** rapid back-and-forth with short turns gives 5.6% confusion and the lowest word speaker accuracy (95.2%).
- **Known ASR misses that affect lexical search:** "pgvector" → "PG vector" (handled for scoring by domain compounds, not in the transcript text), "OpenAI and Anthropic" → "OpenAnthropic" (`audio_04`).

### Change vs the previous MLX run

Standard WER (clean mean) moved from 7.14% to **6.62%**; per file 4.58 → 3.56, 9.48 → 9.16, 9.22 → 9.98, 12.48 → 12.67, 3.15 → 1.08, 3.96 → 3.28, cafe 5.05 → 4.27. Earlier word speaker accuracy figures are not directly comparable (they used first-token-only matching and difflib autojunk).

### Runtime (Apple M4, 10 CPU cores, per ~9-minute file)

| ASR | Alignment | Diarization | Total |
| :---: | :---: | :---: | :---: |
| 82–97 s (RTF ≈ 0.16) | ~9 s | 227–256 s | **≈ 5.5–6 min** |

Diarization on CPU dominates. Expect Docker on the same machine to be somewhat slower (Linux VM).

---

## 4. Tests

`python -m unittest discover` → **105 tests, all passing.**

| Module | Covers |
| :--- | :--- |
| `test_pipeline_common.py` | cache key sensitivity (config, input bytes, key order), cache validity (missing / corrupt / legacy), CLI parsing, atomic JSON writes, HF token sources and fail-closed |
| `test_ingest.py` | 7 formats → 16 kHz mono PCM16, content-hash id, re-upload reuse, explicit id, too long / too short / unsupported / empty / corrupt / video-without-audio / missing |
| `test_asr.py` | raw artifact contract, cache hit, `--force`, config change invalidation, decode guards |
| `test_align.py` | interpolation (gaps, runs, edges, zero gap, partial timings, regression for the key bug), wildcard detection |
| `test_diarize.py` | RTTM round trip, overlap preservation, centroid refinement (reassign / long-segment guard / separated / insufficient data / non-2 speakers) |
| `test_phase_2_pipeline.py` | canonical contract on all 7 outputs, word assignment, smoothing, turn grouping, short-turn flag, reconcile end to end, speaker accuracy metric |
| `test_qrels_spans.py` | spans inside turns, narrower than long turns, reference-alignment provenance |
| `test_build_dataset.py` | SNR mixing math, committed audio matches `sources.json` |
| `test_eval_metrics.py` (Phase 1) | retrieval metric and gate logic |

---

## 5. Open Items and Limitations

1. **Ground-truth turn timestamps (resolved).** Turn text comes from the official published transcripts. The Lex Fridman pages publish per-paragraph timestamps; the Postgres FM, Changelog, Dwarkesh and Conversations with Tyler pages do not publish per-turn times, so those turns were timed during curation. The curator confirms no Whisper or pyannote output was used, so the references are independent of the pipeline being scored.
2. **Golden audio cannot be rebuilt from source yet.** `sources.json` lacks source audio URLs, clip offsets and audio licenses (`null`); `--verify` works, `--build` skips every file.
3. **Refinement thresholds tuned on the evaluation set** (see §3).
4. **Exactly two speakers assumed.** A third voice (e.g. an ad read) is folded into one of the two.
5. **Backchannels:** exclusive diarization gives each moment to one speaker, and Whisper often omits "mm-hmm"/"yeah", so many backchannels do not appear at all.
6. **CPU diarization is slow** (~4 min per 9-minute file); relevant for the upload UI's progress display.

---

## 6. Handoff to Phase B / Phase 3

- Canonical transcripts: `dataset/pipeline_outputs/*_canonical.json` (schema in spec §4).
- Speaker names: to be supplied by the Streamlit labeling step and joined at read time; `evals/assertions.py` must resolve `speaker_label` → name before comparing to qrel speakers.
- qrel moments are now tight spans, so the Phase 1 match rule (IoU ≥ 0.3, or start within 5 s with ≥ 25% coverage) should be revisited once Phase 3 chunk sizes are chosen: a 30 s chunk that fully contains a 10 s target has IoU 0.33, which passes only barely.
