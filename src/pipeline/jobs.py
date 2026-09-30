"""
Background job runner for user uploads.

The Streamlit app never runs ML work in its own process. It ingests the upload (fast, gives
immediate validation errors), creates a job file, and launches a detached worker:

    uv run python -m src.pipeline.jobs run <file_id> --workspace data

The worker runs each stage as its own subprocess (the same CLIs as run_pipeline.sh) and records
progress in `<workspace>/jobs/{file_id}.json`; the app only polls that file. Workers take an
exclusive lock so concurrent uploads queue instead of competing for CPU and memory.

A worker touches `<workspace>/jobs/{file_id}.heartbeat` every HEARTBEAT_SECONDS, also while it
waits for the lock. A running job whose newest sign of life (heartbeat or job update) is older
than STALE_SECONDS is marked failed, so a worker that died without saying so (crash, SIGKILL,
a launch that never started, an exited child nobody reaped whose pid still answers, a reused
pid) always ends in a Retry button rather than "Processing" forever.

Status flow: queued → transcribing → aligning → diarizing → reconciling → indexing → awaiting_labels → labeled
             (any running state) → failed
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from src.pipeline.common import Workspace, file_base, list_audio_files, write_json
from src.pipeline.labels import load_labels

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

STAGES = [
    ("transcribing", "src.pipeline.asr"),
    ("aligning", "src.pipeline.align"),
    ("diarizing", "src.pipeline.diarize"),
    ("reconciling", "src.pipeline.reconcile"),
    ("indexing", "src.search.indexer"),
]
# Default cost per stage before any job has finished on this machine: seconds per second of audio
# for the length-dependent stages, flat seconds otherwise. Replaced by observed timings once available.
DEFAULT_STAGE_COST = {"transcribing": 0.18, "aligning": 12.0, "diarizing": 0.47, "reconciling": 3.0, "indexing": 15.0}
PER_AUDIO_SECOND_STAGES = {"transcribing", "diarizing"}
RATE_HISTORY = 5

RUNNING_STATUSES = {"queued"} | {name for name, _ in STAGES}
DONE_STATUSES = {"awaiting_labels", "labeled"}
LOG_TAIL_LINES = 15
HEARTBEAT_SECONDS = 10
STALE_SECONDS = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def job_path(ws: Workspace, file_id: str) -> str:
    return os.path.join(ws.jobs_dir, f"{file_base(file_id)}.json")


def log_path(ws: Workspace, file_id: str) -> str:
    return os.path.join(ws.jobs_dir, f"{file_base(file_id)}.log")


def heartbeat_path(ws: Workspace, file_id: str) -> str:
    return os.path.join(ws.jobs_dir, f"{file_base(file_id)}.heartbeat")


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _update(ws: Workspace, file_id: str, **fields: Any) -> Dict[str, Any]:
    job = _read_json(job_path(ws, file_id)) or {"file_id": file_id, "created_at": _now(), "stages": {}}
    job.update(fields)
    job["updated_at"] = _now()
    write_json(job_path(ws, file_id), job)
    return job


def create_job(ws: Workspace, file_id: str, wav_path: str, source_filename: str, duration_seconds: float) -> Dict[str, Any]:
    return _update(
        ws,
        file_id,
        wav_path=wav_path,
        source_filename=source_filename,
        duration_seconds=duration_seconds,
        status="queued",
        stages={},
        error=None,
        worker_pid=None,
        launched_at=None,
    )


def _quiet_seconds(ws: Workspace, file_id: str, job: Dict[str, Any]) -> float:
    """Seconds since the worker last showed it was alive: its heartbeat or the job's last update."""
    signs = [datetime.fromisoformat(job["updated_at"]).timestamp()] if job.get("updated_at") else []
    try:
        signs.append(os.path.getmtime(heartbeat_path(ws, file_id)))
    except OSError:
        pass
    return datetime.now(timezone.utc).timestamp() - max(signs) if signs else float("inf")


def _worker_dead(ws: Workspace, file_id: str, job: Dict[str, Any]) -> bool:
    if job.get("worker_pid") and not _pid_alive(job["worker_pid"]):
        return True
    return _quiet_seconds(ws, file_id, job) >= STALE_SECONDS


def load_job(ws: Workspace, file_id: str) -> Optional[Dict[str, Any]]:
    """Reads a job, marking it failed if its worker died without finishing."""
    job = _read_json(job_path(ws, file_id))
    if job and job.get("status") in RUNNING_STATUSES and _worker_dead(ws, file_id, job):
        job = _update(ws, file_id, status="failed", error="Worker process stopped unexpectedly. Retry the job.")
    return job


def launch_worker(ws: Workspace, file_id: str) -> Dict[str, Any]:
    job = load_job(ws, file_id)
    if job is None:
        raise FileNotFoundError(f"No job for {file_id}")
    if job.get("status") in RUNNING_STATUSES and job.get("launched_at"):
        return job

    if os.path.exists(heartbeat_path(ws, file_id)):
        os.remove(heartbeat_path(ws, file_id))
    _update(ws, file_id, status="queued", worker_pid=None, error=None, stages={}, launched_at=_now())
    with open(log_path(ws, file_id), "a", encoding="utf-8") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "src.pipeline.jobs", "run", file_id, "--workspace", ws.root],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    # Only the pid is written here; the worker owns every status transition after launch.
    return _update(ws, file_id, worker_pid=proc.pid)


def stage_command(module: str, ws: Workspace, wav_path: str) -> List[str]:
    # sys.executable is the interpreter uv already resolved for this worker; no second `uv run` needed.
    return [sys.executable, "-u", "-m", module, "--workspace", ws.root, "--file", wav_path]


def _log_tail(ws: Workspace, file_id: str) -> str:
    path = log_path(ws, file_id)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return "".join(f.readlines()[-LOG_TAIL_LINES:]).strip()


def _beat(path: str, stop: threading.Event) -> None:
    while True:
        with open(path, "a"):
            os.utime(path)
        if stop.wait(HEARTBEAT_SECONDS):
            return


def run_job(
    ws: Workspace,
    file_id: str,
    build_command: Callable[[str, Workspace, str], List[str]] = stage_command,
) -> Dict[str, Any]:
    os.makedirs(ws.jobs_dir, exist_ok=True)
    stop = threading.Event()
    heartbeat = threading.Thread(target=_beat, args=(heartbeat_path(ws, file_id), stop), name=f"heartbeat-{file_id}", daemon=True)
    heartbeat.start()
    try:
        return _run_stages(ws, file_id, build_command)
    finally:
        stop.set()
        heartbeat.join()


def _run_stages(ws: Workspace, file_id: str, build_command: Callable[[str, Workspace, str], List[str]]) -> Dict[str, Any]:
    job = _update(ws, file_id, worker_pid=os.getpid())
    wav_path = job["wav_path"]

    with open(os.path.join(ws.jobs_dir, ".worker.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stages = dict(job.get("stages", {}))
        for status, module in STAGES:
            stages[status] = {"started_at": _now()}
            _update(ws, file_id, status=status, stages=stages)
            with open(log_path(ws, file_id), "a", encoding="utf-8") as log:
                log.write(f"\n=== {status} ({module}) ===\n")
                log.flush()
                result = subprocess.run(build_command(module, ws, wav_path), cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT)
            stages[status]["finished_at"] = _now()
            stages[status]["returncode"] = result.returncode
            if result.returncode != 0:
                return _update(
                    ws, file_id, status="failed", stages=stages,
                    error=f"Stage '{status}' failed (exit {result.returncode}).\n{_log_tail(ws, file_id)}",
                )

    final = "labeled" if load_labels(ws, file_id) else "awaiting_labels"
    return _update(ws, file_id, status=final, stages=stages, error=None, worker_pid=None)


def _stage_seconds(info: Dict[str, Any]) -> Optional[float]:
    if not info.get("started_at") or not info.get("finished_at") or info.get("returncode") != 0:
        return None
    return (datetime.fromisoformat(info["finished_at"]) - datetime.fromisoformat(info["started_at"])).total_seconds()


def observed_stage_costs(ws: Workspace) -> Dict[str, float]:
    """Median stage cost over the most recent successful jobs, so estimates match this machine."""
    if not os.path.isdir(ws.jobs_dir):
        return dict(DEFAULT_STAGE_COST)
    finished = []
    for name in os.listdir(ws.jobs_dir):
        if name.endswith(".json"):
            job = _read_json(os.path.join(ws.jobs_dir, name)) or {}
            if job.get("status") in DONE_STATUSES and job.get("duration_seconds"):
                finished.append(job)
    finished.sort(key=lambda j: j.get("updated_at", ""), reverse=True)

    costs = dict(DEFAULT_STAGE_COST)
    for stage in DEFAULT_STAGE_COST:
        samples = []
        for job in finished[:RATE_HISTORY]:
            secs = _stage_seconds(job.get("stages", {}).get(stage, {}))
            if secs is not None:
                samples.append(secs / job["duration_seconds"] if stage in PER_AUDIO_SECOND_STAGES else secs)
        if samples:
            costs[stage] = sorted(samples)[len(samples) // 2]
    return costs


def estimate_stage_seconds(stage: str, audio_seconds: float, costs: Dict[str, float]) -> float:
    return costs[stage] * audio_seconds if stage in PER_AUDIO_SECOND_STAGES else costs[stage]


def mark_labeled(ws: Workspace, file_id: str) -> None:
    if _read_json(job_path(ws, file_id)):
        _update(ws, file_id, status="labeled")


def file_status(ws: Workspace, file_id: str) -> str:
    job = load_job(ws, file_id)
    if job:
        return job["status"]
    if os.path.exists(ws.canonical_path(file_base(file_id))):
        return "labeled" if load_labels(ws, file_id) else "awaiting_labels"
    return "not_processed"


def list_files(ws: Workspace) -> List[Dict[str, Any]]:
    if not os.path.isdir(ws.audio_dir):
        return []
    entries = []
    for wav in list_audio_files(ws.audio_dir):
        file_id = os.path.basename(wav)
        job = load_job(ws, file_id) or {}
        entries.append({
            "file_id": file_id,
            "wav_path": wav,
            "display_name": job.get("source_filename") or file_id,
            "status": file_status(ws, file_id),
            "created_at": job.get("created_at"),
        })
    return sorted(entries, key=lambda e: (e["created_at"] or "", e["file_id"]), reverse=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline job worker")
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("file_id")
    run_parser.add_argument("--workspace", default="data")
    args = parser.parse_args()
    final_job = run_job(Workspace(args.workspace), args.file_id)
    sys.exit(0 if final_job["status"] in DONE_STATUSES else 1)
