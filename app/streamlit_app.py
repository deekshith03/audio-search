"""
Audio Search: search what was said across recordings (file, speaker, timestamp, playable clip), or
upload a two-speaker recording, let the pipeline transcribe, diarize and index it, then name each
speaker. Run with:  uv run streamlit run app/streamlit_app.py
"""

import json
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone

import psycopg2
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.db.connection import connect  # noqa: E402
from src.pipeline import jobs  # noqa: E402
from src.pipeline.common import GOLDEN, Workspace, file_base  # noqa: E402
from src.pipeline.ingest import MAX_DURATION_SECONDS, SUPPORTED_EXTENSIONS, IngestError, format_duration, ingest  # noqa: E402
from src.pipeline.labels import LabelValidationError, load_labels, save_labels, speaker_name, swap_labels  # noqa: E402
from src.pipeline.speaker_samples import extract_clip, select_speaker_samples, speaking_time  # noqa: E402
from src.search.engine import SearchEngine, SearchFilters  # noqa: E402
from src.search.indexer import Indexer  # noqa: E402

UPLOADS = Workspace(os.environ.get("APP_DATA_DIR", "data"))
GOLDEN_SET = Workspace(os.environ.get("GOLDEN_DATA_DIR", GOLDEN.root))
WORKSPACES = {"Uploads": UPLOADS, "Golden set": GOLDEN_SET}
VIEWS = ("🔍 Search", "🎙️ Recordings")
SEARCH_MODES = {"Hybrid": "hybrid", "Keyword": "lexical", "Semantic": "dense"}
CLIP_PADDING_SECONDS = 2.0
DB_DOWN = "The search database is not reachable. Start it with `docker compose up -d db`."
STAGE_LABELS = {
    "transcribing": "Transcribing speech",
    "aligning": "Aligning word timings",
    "diarizing": "Identifying speakers",
    "reconciling": "Building transcript",
    "indexing": "Indexing for search",
}
STATUS_BADGES = {
    "queued": "⏳ Queued",
    "transcribing": "⚙️ Processing",
    "aligning": "⚙️ Processing",
    "diarizing": "⚙️ Processing",
    "reconciling": "⚙️ Processing",
    "indexing": "⚙️ Processing",
    "awaiting_labels": "🏷️ Needs speaker names",
    "labeled": "✅ Ready",
    "failed": "❌ Failed",
    "not_processed": "○ Not processed",
}


def _elapsed(iso: str) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds()


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def safe_filename(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(name)).strip("._") or "upload"
    return f"{uuid.uuid4().hex[:8]}_{base}"


def select_file(file_id: str) -> None:
    st.session_state["selected"] = file_id


def render_sidebar() -> Workspace:
    ws_name = st.sidebar.radio("Workspace", list(WORKSPACES), horizontal=True, key="workspace_name")
    ws = WORKSPACES[ws_name]

    if ws is UPLOADS and st.sidebar.button("➕ New upload", width="stretch"):
        st.session_state["selected"] = None

    files = jobs.list_files(ws)
    if not files:
        st.sidebar.caption("No recordings yet.")
    for entry in files:
        is_selected = st.session_state.get("selected") == entry["file_id"]
        st.sidebar.button(
            f"{entry['display_name']} · {STATUS_BADGES.get(entry['status'], entry['status'])}",
            key=f"pick_{ws_name}_{entry['file_id']}",
            on_click=select_file,
            args=(entry["file_id"],),
            type="primary" if is_selected else "secondary",
            width="stretch",
        )
    return ws


def render_upload(ws: Workspace) -> None:
    st.header("Upload a conversation")
    st.write(
        f"Two-speaker recordings up to **{format_duration(MAX_DURATION_SECONDS)}** minutes. "
        f"Formats: {', '.join(sorted(e.lstrip('.') for e in SUPPORTED_EXTENSIONS))}."
    )
    uploaded = st.file_uploader("Audio file", type=sorted(e.lstrip(".") for e in SUPPORTED_EXTENSIONS))
    if uploaded is None or not st.button("Process recording", type="primary"):
        return

    os.makedirs(ws.uploads_dir, exist_ok=True)
    src_path = os.path.join(ws.uploads_dir, safe_filename(uploaded.name))
    with open(src_path, "wb") as f:
        f.write(uploaded.getbuffer())

    try:
        with st.spinner("Checking and converting audio..."):
            result = ingest(src_path, out_dir=ws.audio_dir)
    except IngestError as e:
        os.remove(src_path)
        st.error(f"Could not use this file: {e}")
        return

    status = jobs.file_status(ws, result.file_id)
    if status in jobs.RUNNING_STATUSES | jobs.DONE_STATUSES:
        st.session_state["selected"] = result.file_id
        st.rerun()

    jobs.create_job(ws, result.file_id, result.wav_path, uploaded.name, result.duration_seconds)
    jobs.launch_worker(ws, result.file_id)
    st.session_state["selected"] = result.file_id
    st.rerun()


def render_status_caption(status: str, job: dict) -> None:
    meta = [STATUS_BADGES.get(status, status)]
    if job.get("duration_seconds"):
        meta.append(mmss(job["duration_seconds"]))
    st.caption(" · ".join(meta))


@st.fragment(run_every=2)
def render_progress(ws: Workspace, file_id: str) -> None:
    job = jobs.load_job(ws, file_id)
    if job is None or job["status"] not in jobs.RUNNING_STATUSES:
        st.rerun()
        return

    render_status_caption(job["status"], job)

    duration = float(job.get("duration_seconds") or 0.0)
    costs = jobs.observed_stage_costs(ws)
    total_estimate = sum(jobs.estimate_stage_seconds(s, duration, costs) for s in STAGE_LABELS)
    done_estimate = 0.0
    remaining = 0.0
    for stage, label in STAGE_LABELS.items():
        info = job.get("stages", {}).get(stage, {})
        expected = jobs.estimate_stage_seconds(stage, duration, costs)
        if info.get("finished_at"):
            st.markdown(f"✅ {label}")
            done_estimate += expected
        elif info.get("started_at"):
            spent = _elapsed(info["started_at"])
            st.markdown(f"⏳ **{label}**, {mmss(spent)} elapsed")
            done_estimate += min(spent, expected * 0.95)
            remaining += max(expected - spent, 5.0)
        else:
            st.markdown(f"○ {label}")
            remaining += expected

    if job["status"] == "queued":
        st.caption("Waiting for the worker (another recording may be processing).")
    st.progress(min(done_estimate / total_estimate, 0.99) if total_estimate else 0.0)
    st.caption(f"About {max(1, round(remaining / 60))} min remaining. You can leave this page; processing continues.")


def render_failed(ws: Workspace, file_id: str, job: dict) -> None:
    st.error("Processing failed.")
    if job.get("error"):
        st.code(job["error"], language="text")
    if st.button("Retry", type="primary"):
        jobs.launch_worker(ws, file_id)
        st.rerun()


def _load_canonical(ws: Workspace, file_id: str) -> dict:
    with open(ws.canonical_path(file_base(file_id)), "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(max_entries=64, show_spinner=False)
def clip_bytes(wav_path: str, start: float, end: float, mtime: float) -> bytes:
    return extract_clip(wav_path, start, end)


def _swap_names(keys: list) -> None:
    names = swap_labels({k: st.session_state.get(k, "") for k in keys})
    for k, v in names.items():
        st.session_state[k] = v


def render_labeling(ws: Workspace, file_id: str) -> None:
    canonical = _load_canonical(ws, file_id)
    labels_doc = load_labels(ws, file_id)
    wav_path = os.path.join(ws.audio_dir, file_id)
    speakers = canonical["speaker_labels"]
    samples = select_speaker_samples(canonical)
    talk = speaking_time(canonical)
    total_talk = sum(talk.values()) or 1.0

    st.subheader("Who is speaking?")
    st.caption("Listen to a few clips of each voice, then enter their names.")

    keys = [f"name_{file_id}_{s}" for s in speakers]
    for key, speaker in zip(keys, speakers):
        if key not in st.session_state:
            st.session_state[key] = speaker_name(labels_doc, speaker) if labels_doc else ""

    columns = st.columns(len(speakers))
    for col, key, speaker in zip(columns, keys, speakers):
        with col:
            with st.container(border=True):
                st.markdown(f"**Voice {speakers.index(speaker) + 1}** · {mmss(talk[speaker])} spoken ({talk[speaker] / total_talk:.0%})")
                for clip in samples[speaker]:
                    st.caption(f"{mmss(clip['start_seconds'])}–{mmss(clip['end_seconds'])}")
                    st.audio(clip_bytes(wav_path, clip["start_seconds"], clip["end_seconds"], os.path.getmtime(wav_path)), format="audio/wav")
                    st.markdown(f"> {clip['text']}")
                st.text_input("Name", key=key, placeholder="e.g. Lex Fridman")

    left, right = st.columns([1, 1])
    left.button("⇄ Swap names", on_click=_swap_names, args=(keys,), width="stretch")
    if right.button("Save names", type="primary", width="stretch"):
        try:
            save_labels(
                ws, file_id, {s: st.session_state[k] for s, k in zip(speakers, keys)}, speakers,
                labeled_by="golden_simulated" if ws is GOLDEN_SET else "user",
            )
        except LabelValidationError as e:
            st.error(str(e))
        else:
            jobs.mark_labeled(ws, file_id)
            sync_search_speakers(ws, file_id)
            st.rerun()


def sync_search_speakers(ws: Workspace, file_id: str) -> None:
    try:
        conn = connect(connect_timeout=3)
        try:
            Indexer(conn, ws).sync_speaker_names(file_id)
        finally:
            conn.close()
    except psycopg2.Error:
        st.session_state["flash"] = ("warning", "Speaker names saved, but search could not be updated (database not reachable). "
                                                "They sync the next time this recording is indexed.")
    else:
        st.session_state["flash"] = ("success", "Speaker names saved.")


def render_flash() -> None:
    flash = st.session_state.pop("flash", None)
    if flash:
        getattr(st, flash[0])(flash[1])


def render_transcript(ws: Workspace, file_id: str) -> None:
    canonical = _load_canonical(ws, file_id)
    labels_doc = load_labels(ws, file_id)
    with st.expander("Transcript", expanded=bool(labels_doc)):
        for t in canonical["turns"]:
            name = speaker_name(labels_doc, t["speaker_label"])
            st.markdown(f"`{mmss(t['start_seconds'])}` **{name}:** {t['text']}")


def render_file(ws: Workspace, file_id: str) -> None:
    job = jobs.load_job(ws, file_id) or {}
    status = jobs.file_status(ws, file_id)
    st.header(job.get("source_filename") or file_id)

    if status in jobs.RUNNING_STATUSES:
        render_progress(ws, file_id)
        return

    render_status_caption(status, job)
    render_flash()
    if status == "failed":
        render_failed(ws, file_id, job)
    elif status in jobs.DONE_STATUSES:
        render_labeling(ws, file_id)
        render_transcript(ws, file_id)
    else:
        st.info("This recording has not been processed.")


@st.cache_resource(show_spinner=False)
def search_engine() -> SearchEngine:
    return SearchEngine()


@st.cache_resource(show_spinner=False)
def search_lock() -> threading.Lock:
    """The engine holds one connection, so browser sessions take turns using it."""
    return threading.Lock()


def workspace_at(root: str) -> Workspace:
    return next(ws for ws in WORKSPACES.values() if ws.root == root)


def recording_names(rows: list) -> dict:
    """(workspace, file_id) → the name people know the recording by: the uploaded filename, else the file id."""
    uploaded = {}
    for root in {r["workspace"] for r in rows}:
        uploaded.update({(root, e["file_id"]): e["display_name"] for e in jobs.list_files(workspace_at(root)) if e["display_name"] != e["file_id"]})
    return {(r["workspace"], r["file_id"]): uploaded.get((r["workspace"], r["file_id"])) or file_base(r["file_id"]) for r in rows}


def filter_options(rows: list, names: dict) -> tuple:
    """Recording choices keyed by files.id, and speaker choices by name spanning every recording they appear in."""
    recordings, speakers = {}, {}
    for r in rows:
        recording = names[(r["workspace"], r["file_id"])]
        recordings[r["file_pk"]] = recording
        speaker = r["display_name"] or f"{r['speaker_label']} ({recording})"
        speakers.setdefault(speaker, []).append((r["file_pk"], r["speaker_label"]))
    return recordings, speakers


def render_result(r: dict, name: str) -> None:
    wav_path = os.path.join(workspace_at(r["workspace"]).audio_dir, r["file_id"])
    with st.container(border=True):
        st.markdown(f"**{r['rank']}. {name}** · {r['speaker']} · `{mmss(r['start_seconds'])}–{mmss(r['end_seconds'])}`")
        st.html(f"<p>{r['highlight']}</p>")
        if os.path.exists(wav_path):
            clip = clip_bytes(wav_path, r["start_seconds"], r["end_seconds"] + CLIP_PADDING_SECONDS, os.path.getmtime(wav_path))
            st.audio(clip, format="audio/wav")


def render_search() -> None:
    st.header("Search conversations")
    query = st.text_input("Search", key="query", placeholder="e.g. why were arrays added to postgres", label_visibility="collapsed")
    left, middle, right = st.columns([2, 2, 1])
    mode = left.radio("Mode", list(SEARCH_MODES), horizontal=True, key="search_mode")
    with middle:
        st.caption("Search in")
        in_golden = st.checkbox("Golden set", value=True, key="search_golden")
        in_uploads = st.checkbox("Uploads", value=False, key="search_uploads")
    top_k = right.number_input("Results", min_value=1, max_value=20, value=5, key="top_k")
    roots = [ws.root for ws, on in ((GOLDEN_SET, in_golden), (UPLOADS, in_uploads)) if on]
    if not roots:
        st.info("Choose at least one place to search.")
        return

    engine, lock = search_engine(), search_lock()
    try:
        with lock:
            rows = engine.speakers_in(roots)
    except psycopg2.OperationalError:
        st.error(DB_DOWN)
        return
    if not rows:
        st.info("Nothing is indexed here yet. Golden set: `uv run python -m src.search.indexer`; uploads are indexed after processing.")
        return

    names = recording_names(rows)
    recordings, speakers = filter_options(rows, names)
    with st.expander("Filters"):
        picked_files = st.multiselect("Recording", list(recordings), format_func=recordings.get, key="filter_recordings")
        picked_speakers = st.multiselect("Speaker", sorted(speakers), key="filter_speakers")
    if not query.strip():
        return

    filters = SearchFilters(
        file_pks=tuple(picked_files),
        speakers=tuple(pair for name in picked_speakers for pair in speakers[name]),
    )
    try:
        with st.spinner("Searching..."), lock:
            response = engine.search(query, SEARCH_MODES[mode], int(top_k), workspaces=roots, filters=filters)
    except psycopg2.OperationalError:
        st.error(DB_DOWN)
        return

    if not response.results:
        st.info("No matches. Try other words, or Semantic mode for a description of what was said.")
        return
    st.caption(f"{len(response.results)} results · {response.timings_ms.get('total', 0):.0f} ms")
    for r in response.results:
        render_result(r, names.get((r["workspace"], r["file_id"]), file_base(r["file_id"])))


def main() -> None:
    st.set_page_config(page_title="Audio Search", page_icon="🎙️", layout="wide")
    st.sidebar.title("🎙️ Audio Search")
    if st.sidebar.radio("View", VIEWS, horizontal=True, key="view", label_visibility="collapsed") == VIEWS[0]:
        render_search()
        return
    ws = render_sidebar()
    selected = st.session_state.get("selected")
    known = {e["file_id"] for e in jobs.list_files(ws)}
    if selected and selected in known:
        render_file(ws, selected)
    elif ws is UPLOADS:
        render_upload(ws)
    else:
        st.header("Golden set")
        st.write("Select a recording in the sidebar to name its speakers.")


main()
