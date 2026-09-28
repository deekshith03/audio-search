# Phase B Summary: Upload & Speaker Labeling (Streamlit)

A local web app where a user uploads a two-speaker recording, watches it being processed, listens to sample clips of each voice, and names the speakers. The named transcript is the input to Phase 3 search.

```
streamlit run app/streamlit_app.py        # http://localhost:8501
```

---

## 1. Flow

```
 Upload page                Processing page               Labeling page
 ┌──────────────────┐       ┌────────────────────────┐    ┌──────────────────────────────┐
 │ file uploader    │       │ ✅ Transcribing speech  │    │ Voice 1 · 2:46 spoken (33%)  │
 │ (9 formats,      │──────►│ ⏳ Identifying speakers │───►│  ▶ 0:00–0:14  "Hello, ..."   │
 │  ≤ 10 min)       │ ingest│ ○ Building transcript   │    │  ▶ 3:40–3:46  ...            │
 └──────────────────┘ +job  │ progress · ~N min left  │    │  Name [ Tyler Cowen     ]    │
        │ rejects with      └────────────────────────┘    │ Voice 2 ...                  │
        │ a clear reason       polls job file every 2 s    │ [⇄ Swap]  [Save names]       │
        ▼                                                  └──────────────┬───────────────┘
 data/audio/upload_<sha16>.wav                                            ▼
                                       data/speaker_labels/upload_<sha16>.json → ✅ Ready
                                       transcript shown with real names
```

1. **Upload:** the file is saved to `data/uploads/`, validated and converted by `ingest.py` in the request (a few seconds), so format, length and corrupt-file errors appear immediately. Re-uploading the same bytes reuses the earlier results.
2. **Processing:** the app writes `data/jobs/{id}.json` and launches a detached worker (`python -m src.pipeline.jobs run {id} --workspace data`). The worker runs ASR → align → diarize → reconcile as separate subprocesses and updates the job file; the app only reads it. The page can be closed; processing continues.
3. **Labeling:** three 4–15 s clips per voice, spread across the recording and chosen by word confidence, with their text and each voice's share of speaking time. Names are validated (both filled in, different, ≤ 80 characters); **Swap** fixes the common "typed them the wrong way round" case. Names can be changed later without re-processing.

## 2. Status Model

```
queued → transcribing → aligning → diarizing → reconciling → awaiting_labels → labeled
   └───────────────── any running state ─────────────────→ failed (Retry)
```

- A worker holds an exclusive lock, so a second upload waits in `queued` instead of competing for CPU and memory.
- If a worker dies mid-run, the next status read marks the job `failed` ("stopped unexpectedly"); **Retry** relaunches it and cached stages are skipped.
- Golden-set files have no job file; their status is derived from their artifacts (canonical transcript present → `awaiting_labels`, labels present → `labeled`).

## 3. Workspaces

Uploads and the golden set share one directory layout, so every stage works on either:

```
<workspace>/audio/            normalized WAVs (+ .ingest.json for uploads)
<workspace>/cache/raw_asr/    ASR + alignment artifacts
<workspace>/cache/raw_diarization/
<workspace>/pipeline_outputs/ canonical transcripts
<workspace>/speaker_labels/   human names per file
<workspace>/jobs/             job status + worker logs (uploads only)
<workspace>/uploads/          original uploaded files
```

`data/` (uploads, gitignored) and `dataset/` (golden set). Every stage CLI takes `--workspace`. The sidebar switches between **Uploads** and **Golden set**; labels saved in the golden set are recorded as `labeled_by: "golden_simulated"`.

Evaluation joins names at scoring time: `evals/assertions.py` resolves a result's anonymous `SPEAKER_xx` label through `dataset/speaker_labels/` before comparing it with the qrel speaker.

## 4. End-to-End Verification

An 8:26 MP3 (audio_06 re-encoded, so a new content hash and a full pipeline run) was uploaded through the browser UI:

| Stage | Time |
| :--- | :---: |
| Transcribing | 91 s |
| Aligning | 14 s |
| Identifying speakers | 262 s |
| Building transcript | < 1 s |
| **Total** | **≈ 6 min 7 s** |

The page switched to labeling automatically; after naming the voices ("Tyler Cowen", "Cass Sunstein") the status became Ready and the transcript showed the names with correct attribution.

The run exposed one bug, now fixed with a regression test: `ingest()` returned file ids without the `.wav` extension, while every other component (golden set, qrels, canonical files, the app's file list) uses `<name>.wav`, so the app could not find the upload it had just created.

## 5. Tests

159 tests pass (`python -m unittest discover`). New in Phase B:

| Module | Covers |
| :--- | :--- |
| `test_jobs.py` | all four stages run in order and finish in `awaiting_labels`; `labeled` when names already exist; failure stops the pipeline with a log tail; retry; dead-worker detection; no double launch; derived status and listing; ingest → job id contract |
| `test_labels.py` | trimming, empty / duplicate (case-insensitive) / overlong names, wrong speaker set, round trip, relabel, provenance values, invalid labels never written, swap |
| `test_speaker_samples.py` | 3 clips per voice spread over the recording, word-bounded 4–15 s windows, confidence preference, short-turn exclusion, fallback for quiet speakers, speaking time, clip extraction |
| `test_assertions_speaker.py` | anonymous label → name resolution, pass-through of real names, unlabeled files, assertion passes only with the correctly resolved speaker |
| `test_streamlit_app.py` | Streamlit `AppTest`: upload page, labeling view (clips + two name fields), saving writes golden-simulated labels, duplicate-name error saves nothing, swap, processing checklist, failed view with Retry |
| `test_pipeline_common.py` | workspace layout and `--workspace` CLI |

## 6. Limitations

- **One worker at a time.** Uploads are processed sequentially (≈ 6 min each on CPU).
- **Time remaining is an estimate** from measured per-second rates, not real progress from inside each model.
- **Exactly two voices.** A third voice is merged into one of the two; the UI cannot split or merge speakers.
- **Local, single user.** No authentication, deletion or storage limits; original uploads are kept in `data/uploads/`.

## 7. Still To Do (you)

Label the 7 golden files: **Golden set** workspace → each file → name the voices by listening. This creates `dataset/speaker_labels/*.json`, which the retrieval evaluation needs to score speaker attribution.
