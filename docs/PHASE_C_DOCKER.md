# Phase C: Docker (single-command setup)

```bash
cp env.example .env          # set HF_TOKEN
docker compose up --build    # → http://localhost:8501
```

---

## 1. Design

```
docker compose up
   │
   ├── db    pgvector/pgvector:pg16          healthcheck: pg_isready
   │         volume pgdata                   port 127.0.0.1:5433 (idle until Phase 3)
   │
   └── app   audio-search (python:3.12-slim-bookworm + ffmpeg + uv)
             entrypoint: bootstrap models → streamlit :8501
             ├─ volume models    → /models   (HF_HOME, TORCH_HOME, bootstrap marker)
             ├─ volume app_data  → /app/data (uploads workspace)
             └─ background workers run inside the same container (subprocess per stage)
```

| Decision | Why |
| :--- | :--- |
| **CPU-only PyTorch on Linux** (`[tool.uv.sources]` → `download.pytorch.org/whl/cpu`, Linux marker only) | PyPI's Linux torch wheels pull 15 NVIDIA CUDA packages (several GB). macOS keeps the PyPI build; one `uv.lock` serves both. |
| **Dependencies installed from `uv.lock` (`uv sync --frozen`) in their own layer** | Reproducible; code changes rebuild in seconds. |
| **Models downloaded at first start, not baked into the image** | The pyannote model is gated behind its own license and needs the user's token; a volume keeps later starts offline and fast. |
| **Bootstrap marker (`/models/.bootstrap.json`)** | Records the model ids; restarts skip the network entirely. Changing a model id in code triggers a fresh download. |
| **Specific bootstrap exit codes and messages** | Missing token (2), rejected token (3), terms not accepted (4) each print the exact fix and link. |
| **No restart policy on `app`** | A configuration error should stop with its message visible, not loop. |
| **Database bound to 127.0.0.1 with dev credentials** | Local-only; overridable from `.env`. |

### Commands (`scripts/docker-entrypoint.sh`)

| Command | Does |
| :--- | :--- |
| `docker compose up` | bootstrap models (first time), serve the app |
| `docker compose run --rm app reproduce [--recompute]` | provenance check → dataset integrity → scorecard (or full re-run) → unit tests |
| `docker compose run --rm app test` | unit tests |
| `docker compose run --rm app pipeline --file dataset/audio/<f>.wav [--force]` | pipeline stages on golden files |
| `docker compose run --rm app bootstrap` | download models only |
| `docker compose run --rm app bash` | shell |

---

## 2. Verification (Apple M4, Colima VM 8 CPU / 10 GiB, linux/arm64)

| Check | Result |
| :--- | :--- |
| `docker compose build` from scratch | ✓ image 3.85 GB (venv 2.1 GB, of which torch 651 MB) |
| `docker compose run --rm app test` | ✓ 171 tests pass (including ffmpeg ingest and Streamlit AppTest suites) |
| `docker compose run --rm app reproduce` | ✓ provenance 7/7, integrity zero defects, scorecard identical to native, tests pass |
| `docker compose up`, first start | ✓ pyannote → Whisper → wav2vec2 downloaded (1.9 GB volume), app healthy |
| `docker compose up`, restart | ✓ "Models already downloaded", healthy in ~10 s |
| Upload an **M4A** (AAC) through the browser | ✓ ingest → 4 stages → labeling → names saved → Ready |
| Same golden WAV, ASR in Docker vs native | ✓ **byte-identical transcript** (WER 9.16% both) |

### Performance

9:19 recording (`audio_02`):

| Stage | Native macOS | Docker (Linux arm64 VM) |
| :--- | :---: | :---: |
| Transcribing | 97 s | 287–311 s |
| Aligning | 9 s | 36 s |
| Identifying speakers | 252 s | 278 s |
| **Total** | **≈ 6 min** | **≈ 10.4 min** |

ASR is about 3x slower in the container, while diarization is almost unchanged. The most likely cause is that CTranslate2's macOS build uses Apple's Accelerate framework (backed by the M-series matrix units), which a Linux VM cannot reach; PyTorch-based stages do not depend on it. Output is unaffected (identical transcripts). The progress page calibrates its time estimate from the median of recent completed jobs on the same machine, so estimates match whichever environment is running.

Peak memory during ASR: ≈ 3.6–4.0 GiB (container idle: ≈ 170 MiB). Give Docker at least 8 GB.

---

## 3. Bugs Found During Verification (fixed)

1. **Stale status badge**: the header showed "Queued" while the checklist showed a running stage, because only the checklist was inside the auto-refreshing fragment. The badge now renders inside it (test added).
2. **Time estimate assumed native speed**: in Docker the first estimate was about 3x too optimistic. Estimates now come from observed timings (`jobs.observed_stage_costs`, tests added).

## 4. Limitations

- **CPU only.** No GPU path in the container (by design: one backend everywhere).
- **Golden-set labels edited in the container are not persisted** (`dataset/` is part of the image); the committed labels are what evaluation uses.
- **Image size 3.85 GB**, dominated by PyTorch and the scientific Python stack.
- **Transcript casing**: with `condition_on_previous_text=False`, Whisper emits many segments in lowercase without punctuation (87 of 110 in `audio_02`, identical natively). Full-text search is case-insensitive, so retrieval is unaffected; display readability is.
