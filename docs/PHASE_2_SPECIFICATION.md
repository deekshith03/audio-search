# Phase 2 Technical Specification: Local ML Conversion & Diarization Pipeline

---

## 1. Executive Summary & Objective

Phase 2 implements the **Machine Learning Conversion Pipeline** that transforms the 7 standardized audio files (`dataset/audio/*.wav`) into canonical, speaker-attributed, turn-by-turn transcripts (`dataset/pipeline_outputs/*.json`).

The resulting transcripts serve as the input for **Phase 3 (PostgreSQL + pgvector Hybrid Search Engine)**.

---

## 2. Hardware Architecture & Framework Selection (Apple Silicon M-Series)

To achieve state-of-the-art accuracy and maximum throughput on macOS Apple Silicon:

| Stage | Selected Model & Tool | Execution Target | Rationale |
| :--- | :--- | :--- | :--- |
| **ASR (Primary)** | `whisper-large-v3-turbo` via `mlx-whisper` | **Apple Metal GPU (`Device(gpu, 0)`)** | Apple's native MLX framework runs directly on the unified memory GPU. 4x–8x faster than CPU-only CTranslate2. |
| **ASR (Ablation)** | `parakeet-tdt-0.6b-v2` via `parakeet-mlx` | **Apple Metal GPU (`Device(gpu, 0)`)** | Evaluated and deferred in favor of primary Whisper Large v3 Turbo on MLX for unified memory stability. |
| **Word Alignment** | `wav2vec2` forced alignment (WhisperX aligner) | **PyTorch MPS / CPU** | Frame-level (~20 ms) phoneme boundary anchoring to correct Whisper's cross-attention drift. |
| **Diarization** | `pyannote/speaker-diarization-community-1` | **PyTorch MPS / CPU** | Latest open-weights diarizer evaluated on VoiceArena. Run with `min_speakers=2, max_speakers=2` and `exclusive_mode=True`. |

> **Process Separation Policy:** To prevent unified memory contention on Apple Silicon, ASR, Alignment, Diarization, and Reconciliation are executed as **separate Python processes** sequentially (releasing GPU VRAM completely between steps).

---

## 3. Four-Stage Pipeline Implementation

```
                      RAW AUDIO (7 WAV Files, ~63 mins)
                                      │
 ═════════════════════════════════════╪═════════════════════════════════════
 STAGE 1: BATCH ASR & FORCED ALIGNMENT (Apple Metal GPU)
 ═════════════════════════════════════╪═════════════════════════════════════
                                      ▼
         Branch A: `mlx_whisper.transcribe` (large-v3-turbo)
         • condition_on_previous_text=False (prevents repetition loops)
         • hallucination_silence_threshold=2.0 (suppresses ghost tokens)
         • NO aggressive denoisers (avoids spectral phoneme distortion)
                                      │
                                      ▼
         Cached Artifact: `dataset/cache/raw_asr/{file_id}_raw.json`
                                      │
                                      ▼
         Wav2Vec2 Forced Alignment:
         • Frame-level (~20 ms) phoneme boundary refinement.
         • Missing Token Fallback: Tokens with digits or symbols ("$5", "N3",
           "1,500", "%") missing phoneme alignments are linearly interpolated
           between neighboring words (or fall back to Whisper timestamps).
         • Telemetry: Logs `alignment_fallback_rate` per file.
                                      │
         [Branch B: Parakeet TDT writes directly to aligned format]
                                      │
                                      ▼
         Cached Artifact: `dataset/cache/raw_asr/{file_id}_aligned.json`
                                      │ (Process exits & Metal memory freed)
 ═════════════════════════════════════╪═════════════════════════════════════
 STAGE 2: SPEAKER DIARIZATION (PyTorch MPS / CPU)
 ═════════════════════════════════════╪═════════════════════════════════════
                                      ▼
         `pyannote.audio` (community-1 pipeline)
         • Constraints: min_speakers=2, max_speakers=2
         • Exclusive Mode: Single active speaker per frame slice.
                                      │
                                      ▼
         Cached Artifact: `dataset/cache/raw_diarization/{file_id}.rttm`
                                      │ (Process exits & PyTorch memory freed)
 ═════════════════════════════════════╪═════════════════════════════════════
 STAGE 3: DETERMINISTIC TURN RECONCILIATION & ENRICHMENT
 ═════════════════════════════════════╪═════════════════════════════════════
                                      ▼
         Reconciliation Engine:
         1. Word Assignment: Assign each word to the speaker whose interval
            maximally overlaps its time span.
         2. Diarization Gaps: Words falling in unassigned gaps inherit the
            nearest speaker segment in time.
         3. Turn Splitting: Trigger new conversational turns on:
            • Speaker Change -> `split_reason: "speaker_change"`
            • Natural Pause (>1.5s) -> `split_reason: "pause"`
         4. Short Turn Flagging: Turns <0.8s matching backchannel lexicons
            ("yeah", "right", "okay", "sure") tagged with `is_short_turn: true`.
         5. Name Mapping (Golden Set Enrichment):
            • Pipeline emits raw: `speaker_label: "SPEAKER_00"`.
            • Enrichment layer maps `speaker_name: "DHH"` via maximal temporal
              overlap with ground truth references.
                                      │
                                      ▼
         OUTPUT: `dataset/pipeline_outputs/{file_id}_canonical.json`
                                      │
 ═════════════════════════════════════╪═════════════════════════════════════
 STAGE 4: PIPELINE EVALUATION (WER, DER & SPEAKER ACCURACY)
 ═════════════════════════════════════╪═════════════════════════════════════
                                      ▼
         • WER (Word Error Rate via jiwer + standardized text normalization).
         • DER (Diarization Error Rate via pyannote.metrics):
           - DER computes the optimal speaker mapping internally, so raw labels
             are scored directly.
           - Scored at both 0 ms collar (strict) and 250 ms collar (forgiving).
         • Word Speaker Accuracy: Aligns hypothesis words with reference words
           and computes the percentage of matched words attributed to the correct speaker
           (timestamp-independent metric directly reflecting search attribution).
         • Per-file reporting (Clean files average vs. `audio_07` Cafe Chatter).
         • Run Manifest (`dataset/pipeline_outputs/pipeline_manifest.json`).
```

---

## 4. Canonical Output Schema (`dataset/pipeline_outputs/*.json`)

Every generated transcript strictly conforms to this contract:

```json
{
  "file_id": "audio_01_lex_dhh_omarchy.wav",
  "pipeline_version": "2.0.0",
  "asr_model": "mlx-community/whisper-large-v3-turbo",
  "diarization_model": "pyannote/speaker-diarization-community-1",
  "alignment_model": "wav2vec2-large-960h",
  "audio_duration_seconds": 584.0,
  "telemetry": {
    "total_words": 1420,
    "fallback_aligned_words": 18,
    "alignment_fallback_rate": 0.0127,
    "total_turns": 25,
    "pause_splits": 4,
    "speaker_change_splits": 21
  },
  "speaker_mapping": {
    "SPEAKER_00": "Lex Fridman",
    "SPEAKER_01": "DHH"
  },
  "turns": [
    {
      "turn_id": 1,
      "speaker_label": "SPEAKER_00",
      "speaker_name": "Lex Fridman",
      "start_seconds": 0.0,
      "end_seconds": 16.12,
      "split_reason": "speaker_change",
      "is_short_turn": false,
      "text": "So some people listening to you right now will say DHH is suffering from the old case of AI psychosis...",
      "words": [
        {
          "word": "So",
          "start_seconds": 0.0,
          "end_seconds": 0.28,
          "confidence": 0.95,
          "timing_source": "wav2vec2_aligned"
        },
        {
          "word": "some",
          "start_seconds": 0.32,
          "end_seconds": 0.54,
          "confidence": 0.98,
          "timing_source": "wav2vec2_aligned"
        }
      ]
    }
  ]
}
```

---

## 5. Source Tree for Phase 2

```
src/
└── pipeline/
    ├── __init__.py
    ├── asr.py                 # mlx-whisper GPU transcription (with Parakeet ablation)
    ├── align.py               # wav2vec2 forced alignment with digit/symbol interpolation
    ├── diarize.py             # pyannote community-1 k=2 exclusive diarization
    ├── reconcile.py           # deterministic word-to-speaker turn reconciliation
    ├── evaluate_pipeline.py   # WER (jiwer), DER (0ms/250ms), and Word Speaker Accuracy
    └── run_pipeline.sh        # multi-process sequential runner with GPU memory release
```
