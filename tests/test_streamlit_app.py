import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.pipeline import jobs
from src.pipeline.common import GOLDEN, Workspace
from src.pipeline.labels import load_labels

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "streamlit_app.py")
GOLDEN_FILE = "audio_06_constitutional_jurisprudence"


class AppTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.uploads = Workspace(os.path.join(self.tmp.name, "data"))
        self.golden = Workspace(os.path.join(self.tmp.name, "golden"))
        os.makedirs(self.golden.audio_dir)
        os.makedirs(self.golden.output_dir)
        os.symlink(os.path.abspath(os.path.join(GOLDEN.audio_dir, f"{GOLDEN_FILE}.wav")), os.path.join(self.golden.audio_dir, f"{GOLDEN_FILE}.wav"))
        shutil.copy(GOLDEN.canonical_path(GOLDEN_FILE), self.golden.canonical_path(GOLDEN_FILE))
        self.env = patch.dict(os.environ, {"APP_DATA_DIR": self.uploads.root, "GOLDEN_DATA_DIR": self.golden.root})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def app(self, **session):
        at = AppTest.from_file(APP, default_timeout=60)
        for k, v in session.items():
            at.session_state[k] = v
        return at

    def open_golden(self):
        at = self.app(selected=f"{GOLDEN_FILE}.wav").run()
        at.radio(key="workspace_name").set_value("Golden set").run()
        return at


class TestUploadView(AppTestCase):

    def test_empty_uploads_workspace_shows_upload_page(self):
        at = self.app().run()
        self.assertFalse(at.exception)
        self.assertIn("Upload a conversation", [h.value for h in at.header])


class TestLabelingView(AppTestCase):

    def name_key(self, speaker):
        return f"name_{GOLDEN_FILE}.wav_{speaker}"

    def test_shows_clips_and_name_fields_per_speaker(self):
        at = self.open_golden()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.text_input), 2)
        self.assertIn("Who is speaking?", [s.value for s in at.subheader])

    def test_saving_names_writes_golden_simulated_labels(self):
        at = self.open_golden()
        at.text_input(key=self.name_key("SPEAKER_00")).input("Tyler Cowen")
        at.text_input(key=self.name_key("SPEAKER_01")).input("Cass Sunstein")
        next(b for b in at.button if b.label == "Save names").click().run()

        doc = load_labels(self.golden, f"{GOLDEN_FILE}.wav")
        self.assertEqual(doc["labels"], {"SPEAKER_00": "Tyler Cowen", "SPEAKER_01": "Cass Sunstein"})
        self.assertEqual(doc["labeled_by"], "golden_simulated")
        self.assertFalse(os.path.exists(os.path.join(GOLDEN.labels_dir, f"{GOLDEN_FILE}.json.tmp")))

    def test_duplicate_names_show_error_and_save_nothing(self):
        at = self.open_golden()
        at.text_input(key=self.name_key("SPEAKER_00")).input("Same")
        at.text_input(key=self.name_key("SPEAKER_01")).input("same")
        next(b for b in at.button if b.label == "Save names").click().run()
        self.assertTrue(any("different name" in e.value for e in at.error))
        self.assertIsNone(load_labels(self.golden, f"{GOLDEN_FILE}.wav"))

    def test_swap_exchanges_entered_names(self):
        at = self.open_golden()
        at.text_input(key=self.name_key("SPEAKER_00")).input("A")
        at.text_input(key=self.name_key("SPEAKER_01")).input("B")
        next(b for b in at.button if b.label == "⇄ Swap names").click().run()
        self.assertEqual(at.text_input(key=self.name_key("SPEAKER_00")).value, "B")
        self.assertEqual(at.text_input(key=self.name_key("SPEAKER_01")).value, "A")


class TestJobViews(AppTestCase):

    def make_job(self, **fields):
        os.makedirs(self.uploads.audio_dir, exist_ok=True)
        wav = os.path.join(self.uploads.audio_dir, "upload_x.wav")
        open(wav, "wb").close()
        jobs.create_job(self.uploads, "upload_x.wav", wav, "talk.mp3", 540.0)
        jobs._update(self.uploads, "upload_x.wav", **fields)

    def test_processing_view_lists_stages(self):
        self.make_job(status="diarizing", worker_pid=os.getpid(), stages={
            "transcribing": {"started_at": "2026-01-01T00:00:00+00:00", "finished_at": "2026-01-01T00:01:30+00:00"},
            "aligning": {"started_at": "2026-01-01T00:01:30+00:00", "finished_at": "2026-01-01T00:01:40+00:00"},
            "diarizing": {"started_at": "2026-01-01T00:01:40+00:00"},
        })
        at = self.app(selected="upload_x.wav").run()
        self.assertFalse(at.exception)
        text = " ".join(m.value for m in at.markdown)
        self.assertIn("✅ Transcribing speech", text)
        self.assertIn("**Identifying speakers**", text)
        self.assertIn("○ Building transcript", text)
        self.assertIn("⚙️ Processing · 9:00", [c.value for c in at.caption])

    def test_failed_view_shows_error_and_retry(self):
        self.make_job(status="failed", error="Stage 'diarizing' failed (exit 1).")
        at = self.app(selected="upload_x.wav").run()
        self.assertTrue(any("Processing failed" in e.value for e in at.error))
        self.assertIn("Retry", [b.label for b in at.button])
        self.assertIn("Stage 'diarizing' failed", " ".join(c.value for c in at.code))


if __name__ == "__main__":
    unittest.main()
