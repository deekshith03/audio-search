import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.pipeline import jobs
from src.pipeline.common import Workspace
from src.pipeline.ingest import ingest
from src.pipeline.labels import save_labels


def ok_command(module, ws, wav_path):
    return [sys.executable, "-c", f"print('ran {module}')"]


def failing_at(stage_module, exit_code=3):
    def build(module, ws, wav_path):
        if module == stage_module:
            return [sys.executable, "-c", f"import sys; print('boom in {module}'); sys.exit({exit_code})"]
        return ok_command(module, ws, wav_path)
    return build


def dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


class JobTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(self.tmp.name)
        os.makedirs(self.ws.audio_dir)
        self.wav = os.path.join(self.ws.audio_dir, "upload_abc.wav")
        open(self.wav, "wb").close()
        jobs.create_job(self.ws, "upload_abc.wav", self.wav, "My Podcast.mp3", 540.0)

    def tearDown(self):
        self.tmp.cleanup()


class TestRunJob(JobTestCase):

    def test_success_walks_every_stage_and_awaits_labels(self):
        job = jobs.run_job(self.ws, "upload_abc.wav", build_command=ok_command)
        self.assertEqual(job["status"], "awaiting_labels")
        self.assertEqual(list(job["stages"]), [s for s, _ in jobs.STAGES])
        for info in job["stages"].values():
            self.assertEqual(info["returncode"], 0)
            self.assertIn("finished_at", info)
        self.assertIsNone(job["worker_pid"])
        with open(jobs.log_path(self.ws, "upload_abc.wav")) as f:
            log = f.read()
        self.assertIn("=== diarizing (src.pipeline.diarize) ===", log)
        self.assertIn("ran src.pipeline.reconcile", log)
        self.assertIn("=== indexing (src.search.indexer) ===", log)

    def test_indexing_runs_after_the_transcript_is_built(self):
        names = [s for s, _ in jobs.STAGES]
        self.assertEqual(names[-2:], ["reconciling", "indexing"])
        self.assertEqual(jobs.stage_command("src.search.indexer", self.ws, "/x/upload.wav")[-4:],
                         ["--workspace", self.ws.root, "--file", "/x/upload.wav"])
        self.assertIn("indexing", jobs.RUNNING_STATUSES)
        self.assertEqual(jobs.estimate_stage_seconds("indexing", 600.0, jobs.DEFAULT_STAGE_COST), jobs.DEFAULT_STAGE_COST["indexing"])

    def test_existing_labels_finish_as_labeled(self):
        save_labels(self.ws, "upload_abc.wav", {"SPEAKER_00": "A", "SPEAKER_01": "B"}, ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(jobs.run_job(self.ws, "upload_abc.wav", build_command=ok_command)["status"], "labeled")

    def test_failed_stage_stops_pipeline_with_log_tail(self):
        job = jobs.run_job(self.ws, "upload_abc.wav", build_command=failing_at("src.pipeline.align"))
        self.assertEqual(job["status"], "failed")
        self.assertIn("Stage 'aligning' failed (exit 3)", job["error"])
        self.assertIn("boom in src.pipeline.align", job["error"])
        self.assertNotIn("diarizing", job["stages"])

    def test_retry_after_failure_succeeds(self):
        jobs.run_job(self.ws, "upload_abc.wav", build_command=failing_at("src.pipeline.asr"))
        job = jobs.run_job(self.ws, "upload_abc.wav", build_command=ok_command)
        self.assertEqual(job["status"], "awaiting_labels")
        self.assertIsNone(job["error"])

    def test_stage_command_targets_workspace_and_file(self):
        cmd = jobs.stage_command("src.pipeline.asr", self.ws, self.wav)
        self.assertEqual(cmd[-6:], ["-m", "src.pipeline.asr", "--workspace", self.ws.root, "--file", self.wav])


class TestJobState(JobTestCase):

    def test_create_job_records_upload_metadata(self):
        job = jobs.load_job(self.ws, "upload_abc.wav")
        self.assertEqual((job["status"], job["source_filename"], job["duration_seconds"]), ("queued", "My Podcast.mp3", 540.0))

    def test_running_job_with_dead_worker_is_marked_failed(self):
        jobs._update(self.ws, "upload_abc.wav", status="diarizing", worker_pid=dead_pid())
        job = jobs.load_job(self.ws, "upload_abc.wav")
        self.assertEqual(job["status"], "failed")
        self.assertIn("stopped unexpectedly", job["error"])

    def test_running_job_with_live_worker_is_left_alone(self):
        jobs._update(self.ws, "upload_abc.wav", status="diarizing", worker_pid=os.getpid())
        self.assertEqual(jobs.load_job(self.ws, "upload_abc.wav")["status"], "diarizing")

    def test_launch_records_pid_and_does_not_double_launch(self):
        fake_proc = MagicMock(pid=os.getpid())
        with patch.object(jobs.subprocess, "Popen", return_value=fake_proc) as popen:
            first = jobs.launch_worker(self.ws, "upload_abc.wav")
            second = jobs.launch_worker(self.ws, "upload_abc.wav")
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(first["worker_pid"], os.getpid())
        self.assertEqual(second["status"], "queued")
        args = popen.call_args.args[0]
        self.assertEqual(args[1:5], ["-m", "src.pipeline.jobs", "run", "upload_abc.wav"])

    def test_launch_requires_existing_job(self):
        with self.assertRaises(FileNotFoundError):
            jobs.launch_worker(self.ws, "missing.wav")

    def test_mark_labeled(self):
        jobs.mark_labeled(self.ws, "upload_abc.wav")
        self.assertEqual(jobs.load_job(self.ws, "upload_abc.wav")["status"], "labeled")


class TestFileStatus(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(self.tmp.name)
        os.makedirs(self.ws.audio_dir)
        for name in ("a.wav", "b.wav"):
            open(os.path.join(self.ws.audio_dir, name), "wb").close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_status_without_job_is_derived_from_artifacts(self):
        self.assertEqual(jobs.file_status(self.ws, "a.wav"), "not_processed")
        os.makedirs(self.ws.output_dir)
        with open(self.ws.canonical_path("a"), "w") as f:
            json.dump({}, f)
        self.assertEqual(jobs.file_status(self.ws, "a.wav"), "awaiting_labels")
        save_labels(self.ws, "a.wav", {"SPEAKER_00": "A", "SPEAKER_01": "B"}, ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(jobs.file_status(self.ws, "a.wav"), "labeled")

    def test_list_files_uses_upload_names_and_newest_first(self):
        jobs.create_job(self.ws, "b.wav", os.path.join(self.ws.audio_dir, "b.wav"), "interview.m4a", 60.0)
        entries = jobs.list_files(self.ws)
        self.assertEqual([e["file_id"] for e in entries], ["b.wav", "a.wav"])
        self.assertEqual(entries[0]["display_name"], "interview.m4a")
        self.assertEqual(entries[1]["display_name"], "a.wav")

    def test_list_files_on_missing_workspace(self):
        self.assertEqual(jobs.list_files(Workspace(os.path.join(self.tmp.name, "nope"))), [])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg not installed")
class TestIngestToJobContract(unittest.TestCase):

    def test_ingested_file_id_is_listed_and_addressable(self):
        with tempfile.TemporaryDirectory() as d:
            ws = Workspace(d)
            src = os.path.join(d, "talk.mp3")
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=11", src], check=True)
            res = ingest(src, out_dir=ws.audio_dir)
            jobs.create_job(ws, res.file_id, res.wav_path, "talk.mp3", res.duration_seconds)

            listed = jobs.list_files(ws)
            self.assertEqual([e["file_id"] for e in listed], [res.file_id])
            self.assertEqual(listed[0]["status"], "queued")
            self.assertEqual(jobs.load_job(ws, res.file_id)["file_id"], res.file_id)


class TestStageCostEstimates(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def finished_job(self, file_id, duration, asr_seconds, align_seconds, updated_at, status="labeled"):
        stages = {
            "transcribing": {"started_at": "2026-01-01T00:00:00+00:00", "finished_at": f"2026-01-01T00:{asr_seconds // 60:02d}:{asr_seconds % 60:02d}+00:00", "returncode": 0},
            "aligning": {"started_at": "2026-01-01T01:00:00+00:00", "finished_at": f"2026-01-01T01:00:{align_seconds:02d}+00:00", "returncode": 0},
        }
        jobs.create_job(self.ws, file_id, "x.wav", "x.mp3", duration)
        jobs._update(self.ws, file_id, status=status, stages=stages)
        path = jobs.job_path(self.ws, file_id)
        with open(path) as f:
            job = json.load(f)
        job["updated_at"] = updated_at
        with open(path, "w") as f:
            json.dump(job, f)

    def test_defaults_without_history(self):
        self.assertEqual(jobs.observed_stage_costs(self.ws), jobs.DEFAULT_STAGE_COST)

    def test_uses_median_of_recent_successful_jobs(self):
        self.finished_job("a.wav", 100.0, 30, 10, "2026-01-03")
        self.finished_job("b.wav", 100.0, 50, 20, "2026-01-02")
        self.finished_job("c.wav", 100.0, 60, 40, "2026-01-01")
        costs = jobs.observed_stage_costs(self.ws)
        self.assertAlmostEqual(costs["transcribing"], 0.5)
        self.assertEqual(costs["aligning"], 20.0)
        self.assertEqual(costs["diarizing"], jobs.DEFAULT_STAGE_COST["diarizing"])

    def test_failed_and_running_jobs_are_ignored(self):
        self.finished_job("a.wav", 100.0, 59, 10, "2026-01-01", status="failed")
        self.assertEqual(jobs.observed_stage_costs(self.ws)["transcribing"], jobs.DEFAULT_STAGE_COST["transcribing"])

    def test_estimate_scales_only_length_dependent_stages(self):
        costs = {"transcribing": 0.5, "aligning": 20.0, "diarizing": 0.4, "reconciling": 2.0}
        self.assertEqual(jobs.estimate_stage_seconds("transcribing", 600, costs), 300.0)
        self.assertEqual(jobs.estimate_stage_seconds("aligning", 600, costs), 20.0)
