# Phase 2 Technical Specification: Local Transcription & Diarization Pipeline

---

## 1. Objective

Phase 2 turns an audio recording into a canonical, word-timed, speaker-attributed transcript
(`dataset/pipeline_outputs/{id}_canonical.json`). These transcripts are the input to Phase 3
(chunking, PostgreSQL + pgvector indexing, hybrid search).

The pipeline serves two callers:
- **Golden set (7 files):** batch run plus quality evaluation against the Phase 1 ground truth.
- **User uploads (Phase B, Streamlit):** one file at a time, followed by a human speaker-labeling step.

---

## 2. Stack

One configuration runs everywhere (native macOS and Docker), so every environment produces the same outputs.

| Stage | Model / Tool | Device | Rationale |
| :--- | :--- | :--- | :--- |
| **Ingest** | `ffprobe` + `ffmpeg` | CPU | Accepts wav/mp3/m4a/aac/flac/ogg/opus/webm/mp4, 10 s to 10 min; outputs 16 kHz mono PCM16. |
| **ASR** | `faster-whisper` `large-v3-turbo`, `compute_type=float32` | CPU | Runs identically in Docker (no Metal/CUDA dependency). Benchmarked against `mlx-whisper` on audio_04/audio_07: equal WER within 0.3–0.8 points, about 2x slower (see §6). |
| **Word alignment** | WhisperX aligner, `WAV2VEC2_ASR_BASE_960H` (torchaudio) | CPU | CTC frame-level (~20 ms) word boundaries. |
| **Diarization** | `pyannote/speaker-diarization-community-1` | CPU | Exactly two speakers (`min_speakers = max_speakers = 2`), exclusive intervals for word assignment. |

Deliberately **not** used:
- **VAD gating (Silero, `vad_filter`):** Whisper's `hallucination_silence_threshold` plus `condition_on_previous_text=False` handle silence hallucinations, and the benchmark (§6) shows low deletion counts without VAD. VAD was not evaluated; it is a candidate if uploads contain long non-speech stretches.
- **Loudness normalization (-23 LUFS):** not evaluated. Whisper log-mel features and pyannote embeddings are fairly level-robust and the golden set is professionally mastered podcast audio; revisit if quiet user uploads show higher WER.
- **ASR model comparison (Parakeet):** out of scope; the only ASR comparison run is faster-whisper vs mlx-whisper (§6).

**Process separation:** each stage runs as its own Python process with its own cached artifact, so a changed or failed stage re-runs in isolation and the upload UI can report per-stage status.

---

## 3. Pipeline

```
 upload / dataset WAV
        │
 STAGE 0  ingest.py        ffprobe validate → ffmpeg 16 kHz mono PCM16 → {file_id}.wav
        │                  file_id = "upload_" + sha256(original)[:16] for uploads
        ▼
 STAGE 1A asr.py           faster-whisper large-v3-turbo fp32 CPU
        │                  language=en, beam_size=5, word_timestamps=True,
        │                  condition_on_previous_text=False, hallucination_silence_threshold=2.0
        ▼                  → cache/raw_asr/{id}_raw.json
 STAGE 1B align.py         WhisperX wav2vec2 forced alignment (CPU)
        │                  • chars outside the vocabulary (digits, $, %) align via a wildcard
        │                    token → counted as wildcard_aligned_words
        │                  • words with no timing at all are spread evenly across the gap
        │                    between timed neighbours → timing_source "interpolated_fallback"
        ▼                  → cache/raw_asr/{id}_aligned.json
 STAGE 2  diarize.py       pyannote community-1, k = 2, CPU
        │                  → {id}.raw.rttm   untouched overlap-aware output
        │                  → {id}.rttm       exclusive intervals after centroid refinement
        ▼                  → {id}_diarization.json (intervals + refinement telemetry)
 STAGE 3  reconcile.py     word → speaker (>50% overlap, else nearest interval)
        │                  isolated-word smoothing (non-cascading), turn splits on speaker
        │                  change / pause > 1.5 s / end of audio, short-turn flag
        ▼                  → pipeline_outputs/{id}_canonical.json
 STAGE 4  evaluate_pipeline.py (golden set only)
                           WER, DER (raw / refined / reconciled), word speaker accuracy
                           → pipeline_outputs/pipeline_manifest.json
```

### Caching

Every artifact stores a `cache_key` = SHA-256 of (stage config + SHA-256 of its inputs). An
artifact is reused only if its key matches, so changing a model, a parameter, or an upstream
artifact invalidates everything downstream. `--force` recomputes regardless.

### Centroid refinement (Stage 2)

When the two speakers' voice centroids (built from segments ≥ 6 s) have cosine similarity
> 0.35, short segments (0.5–5 s) are relabelled if their embedding is closer to the other
speaker by a margin > 0.10. These thresholds were tuned on the golden set (audio_05 rapid
banter), so refined DER on the golden set is optimistic; raw DER is always reported beside it.

### Speaker identity

The pipeline emits only anonymous diarization labels (`SPEAKER_00`, `SPEAKER_01`). Human names
come from a separate labeling step (the user listens to sample snippets and names each speaker)
and are stored outside the transcript, so renaming never requires re-processing or re-indexing.

---

## 4. Canonical Output Schema

```json
{
  "file_id": "audio_01_lex_dhh_omarchy.wav",
  "pipeline_version": "2.1.0",
  "asr_model": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
  "alignment_model": "WAV2VEC2_ASR_BASE_960H",
  "diarization_model": "pyannote/speaker-diarization-community-1",
  "audio_duration_seconds": 584.0,
  "upstream_cache_keys": { "align": "<sha256>", "diarize": "<sha256>" },
  "telemetry": {
    "total_words": 1480,
    "fallback_aligned_words": 0,
    "alignment_fallback_rate": 0.0,
    "wildcard_aligned_words": 12,
    "total_turns": 20,
    "pause_splits": 6,
    "speaker_change_splits": 13
  },
  "speaker_labels": ["SPEAKER_00", "SPEAKER_01"],
  "turns": [
    {
      "turn_id": 1,
      "speaker_label": "SPEAKER_00",
      "start_seconds": 0.0,
      "end_seconds": 16.12,
      "split_reason": "speaker_change",
      "is_short_turn": false,
      "text": "So some people listening to you right now will say ...",
      "words": [
        { "word": "So", "start_seconds": 0.0, "end_seconds": 0.28, "confidence": 0.95, "timing_source": "wav2vec2_aligned" }
      ]
    }
  ]
}
```

- `confidence` is `null` for `interpolated_fallback` words.
- `split_reason` ∈ `speaker_change | pause | end_of_audio`.
- `pause_splits + speaker_change_splits + 1 == total_turns`.

---

## 5. Evaluation (Stage 4)

| Metric | What it measures | Notes |
| :--- | :--- | :--- |
| **Standard WER** | ASR accuracy | OpenAI Whisper English normalizer + domain compounds (`pg vector` → `pgvector`). Published transcripts are lightly edited (fillers, false starts removed), so insertions are partly real speech the reference omits. |
| **Basic WER** | same, lowercase/punctuation only | Legacy comparison. |
| **DER raw** | pyannote output as produced | 0 ms and 250 ms collars, overlap scored. |
| **DER refined** | after centroid refinement | Tuned on this set, so optimistic. |
| **DER reconciled** | final word-bounded turns | Reference turns are gap-free (silence labelled as speech), so every system pays "missed detection" for real pauses; reconciled turns bridge pauses ≤ 1.5 s and look better mostly for that reason. Not a measure of better speaker attribution. |
| **Word speaker accuracy** | share of matched words attributed to the right speaker | Timestamp-independent (difflib token alignment, optimal label mapping computed inside the metric). **Headline speaker metric.** |

The optimal label → reference-speaker mapping is computed only inside scoring and never written back to pipeline outputs.

---

## 6. ASR Backend Benchmark (decision record)

Raw ASR text vs ground truth, standard WER, Apple M4 (10-core CPU / GPU):

| File | Backend | WER | Sub / Del / Ins | ASR time |
| :--- | :--- | :---: | :---: | :---: |
| audio_04 (jargon) | mlx-whisper fp16 (GPU) | 12.48% | 45 / 42 / 111 | 46 s |
| | faster-whisper int8 (CPU) | 13.61% | 45 / 28 / 143 | 134 s |
| | **faster-whisper fp32 (CPU)** | **12.73%** | 41 / 20 / 141 | **101 s** |
| audio_07 (cafe noise) | mlx-whisper fp16 (GPU) | 5.05% | 20 / 59 / 5 | 44 s |
| | faster-whisper int8 (CPU) | 4.57% | 24 / 41 / 11 | 112 s |
| | **faster-whisper fp32 (CPU)** | **4.27%** | 20 / 44 / 7 | **89 s** |

fp32 was chosen: accuracy equal to MLX, faster than int8 on ARM, and it runs unchanged in Docker.

---

## 7. Source Tree

```
src/pipeline/
├── common.py              # cache keys, per-file CLI (--file, --force), HF token loading
├── ingest.py              # Stage 0: ffprobe validation + ffmpeg normalization
├── asr.py                 # Stage 1A: faster-whisper fp32 CPU
├── align.py               # Stage 1B: wav2vec2 alignment + untimed-word interpolation
├── diarize.py             # Stage 2: pyannote k=2, raw + refined RTTM
├── reconcile.py           # Stage 3: word → speaker turns
├── evaluate_pipeline.py   # Stage 4: WER, DER x3, word speaker accuracy, manifest
└── run_pipeline.sh        # runs stages 1A–4 as separate processes; forwards --file/--force
scripts/
├── build_dataset.py       # verify / rebuild golden audio from dataset/metadata/sources.json
└── tighten_qrels.py       # narrow qrel moments to matched_text spans via reference alignment
```

Run: `src/pipeline/run_pipeline.sh [--file dataset/audio/x.wav ...] [--force]`
