# Audio Search

Hybrid (keyword + semantic) search across two-speaker conversation recordings. Upload a recording, the pipeline transcribes it and works out who spoke when, you name the two speakers, and (Phase 3) search returns the file, timestamp and speaker for every hit.

```
upload ─► ingest ─► ASR ─► align ─► diarize ─► reconcile ─► name speakers ─► (Phase 3) index + search
          ffmpeg   faster-   wav2vec2  pyannote   word→speaker   Streamlit
          ≤10 min  whisper   CPU       2 voices   turns          labeling
                   fp32 CPU
```

Everything runs locally on CPU, identically on macOS and in Docker.

---

## Quick start (Docker, one command)

**Prerequisites**

1. Docker with **at least 8 GB of memory** available to containers (Docker Desktop: Settings → Resources; Colima: `colima start --cpu 6 --memory 10`). Whisper large-v3-turbo in fp32 alone needs roughly 3–4 GB.
2. A Hugging Face account:
   - create a read token at <https://huggingface.co/settings/tokens>
   - accept the terms of the gated diarization model at <https://huggingface.co/pyannote/speaker-diarization-community-1>

**Run**

```bash
cp env.example .env        # then set HF_TOKEN=hf_... in .env
docker compose up --build
```

Open <http://localhost:8501>. The first start downloads about 2 GB of models into a Docker volume (a few minutes); later starts are ready in seconds. Processing a 9-minute recording takes about 10 minutes in Docker on an Apple Silicon Mac, or about 6 minutes running natively (see [docs/PHASE_C_DOCKER.md](docs/PHASE_C_DOCKER.md#performance)).

**Reproduce the benchmark numbers**

```bash
docker compose run --rm app reproduce              # verify dataset, score committed outputs, run tests (minutes)
docker compose run --rm app reproduce --recompute  # re-run the whole pipeline on all 7 files first (~70 min in Docker)
```

Other commands: `docker compose run --rm app test`, `docker compose run --rm app pipeline --file dataset/audio/<file>.wav --force`, `docker compose down` (add `-v` to delete models, uploads and the database).

---

## Using the app

1. **Uploads → New upload**: pick a file (wav, mp3, m4a, aac, flac, ogg, opus, webm, mp4; 10 s to 10 min). Invalid files are rejected immediately with the reason.
2. Processing runs in the background with a per-stage progress view; you can close the page.
3. **Who is speaking?**: play three clips per voice, type each person's name, **Save names**. **⇄ Swap names** fixes a mix-up; names can be changed any time without re-processing.
4. The **Golden set** workspace shows the 7 benchmark recordings (already labeled).

---

## Local development (without Docker)

Requires Python 3.12, [uv](https://docs.astral.sh/uv/) and ffmpeg (`brew install ffmpeg`).

```bash
uv sync
cp env.example .env                               # set HF_TOKEN
.venv/bin/python -m scripts.bootstrap_models      # optional: download models up front
.venv/bin/streamlit run app/streamlit_app.py
```

| Task | Command |
| :--- | :--- |
| Unit tests (171) | `.venv/bin/python -m unittest discover` |
| Pipeline on the golden set | `src/pipeline/run_pipeline.sh [--file dataset/audio/x.wav] [--force]` |
| Pipeline on uploads | add `--workspace data` to any stage, e.g. `.venv/bin/python -m src.pipeline.asr --workspace data` |
| Reproduce benchmark | `scripts/reproduce.sh [--recompute]` |
| Verify golden audio | `.venv/bin/python -m scripts.build_dataset --verify` |
| Retrieval eval harness | `npm run eval` (Phase 3+) |

---

## Repository layout

```
app/streamlit_app.py        upload, progress, speaker labeling UI
src/pipeline/               ingest, asr, align, diarize, reconcile, evaluate_pipeline,
                            jobs (background worker), labels, speaker_samples, common
scripts/                    bootstrap_models, reproduce.sh, build_dataset, tighten_qrels, docker-entrypoint.sh
evals/                      promptfoo retrieval harness, recall@k scorer, dataset integrity checks
tests/                      unit + Streamlit AppTest suites
dataset/
  audio/                    7 golden recordings (16 kHz mono WAV, 8–10 min, unique speaker pairs)
  ground_truth/             official transcripts with speaker turns
  qrels/                    18 labeled benchmark queries (29 target moments)
  speaker_labels/           human speaker names for the golden set
  pipeline_outputs/         canonical transcripts + quality manifest
  metadata/sources.json     provenance of every golden file
data/                       uploads workspace (created at runtime, gitignored)
docs/                       phase specifications and summaries
```

## Results so far (golden set, 7 files)

| Metric | Value |
| :--- | :---: |
| Word error rate (mean) | 6.29% |
| Word speaker accuracy (mean) | 98.40% |
| Diarization error rate, raw pyannote (0 ms collar) | 8.68% |

Details, caveats and per-file numbers: [docs/PHASE_2_SUMMARY.md](docs/PHASE_2_SUMMARY.md).

## Documentation

- [docs/PHASE_1_SUMMARY.md](docs/PHASE_1_SUMMARY.md): golden dataset, benchmark queries, evaluation harness
- [docs/PHASE_2_SPECIFICATION.md](docs/PHASE_2_SPECIFICATION.md): pipeline design, schema, ASR backend decision
- [docs/PHASE_2_SUMMARY.md](docs/PHASE_2_SUMMARY.md): what was fixed, quality scorecard, open items
- [docs/PHASE_B_SUMMARY.md](docs/PHASE_B_SUMMARY.md): upload and labeling app
- [docs/PHASE_C_DOCKER.md](docs/PHASE_C_DOCKER.md): container design and verification

## Troubleshooting

| Symptom | Fix |
| :--- | :--- |
| `HF_TOKEN is not set` on start | Set it in `.env`, then `docker compose up` again. |
| `has not accepted the terms` | Accept the conditions at the pyannote model page with the same account as the token. |
| Processing fails with exit code 137 / "Killed" | The container ran out of memory; give Docker at least 8 GB. |
| Port 8501 or 5433 in use | Set `APP_PORT` / `POSTGRES_PORT` in `.env`. |
