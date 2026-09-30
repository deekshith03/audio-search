import json
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import psycopg2
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.pipeline import jobs
from src.pipeline.common import GOLDEN, Workspace
from src.pipeline.labels import load_labels
from src.search.engine import SearchFilters, SearchResponse

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
        session.setdefault("view", "🎙️ Recordings")
        for k, v in session.items():
            at.session_state[k] = v
        return at

    def open_golden(self):
        at = self.app(selected=f"{GOLDEN_FILE}.wav").run()
        at.radio(key="workspace_name").set_value("Golden set").run()
        return at


class FakeEngine:
    """Stands in for SearchEngine: two speakers in one golden recording, one unnamed upload speaker."""

    def __init__(self, golden_root, uploads_root, fail=False):
        self.golden_root, self.uploads_root, self.fail = golden_root, uploads_root, fail
        self.calls = []
        self.result = {}

    def speakers_in(self, workspaces):
        if self.fail:
            raise psycopg2.OperationalError("down")
        rows = [
            {"file_pk": 1, "workspace": self.golden_root, "file_id": f"{GOLDEN_FILE}.wav", "speaker_label": "SPEAKER_00", "display_name": "Tyler Cowen"},
            {"file_pk": 1, "workspace": self.golden_root, "file_id": f"{GOLDEN_FILE}.wav", "speaker_label": "SPEAKER_01", "display_name": "Cass Sunstein"},
            {"file_pk": 2, "workspace": self.uploads_root, "file_id": "upload_x.wav", "speaker_label": "SPEAKER_00", "display_name": None},
            {"file_pk": 3, "workspace": self.golden_root, "file_id": "other.wav", "speaker_label": "SPEAKER_01", "display_name": "Tyler Cowen"},
        ]
        return [r for r in rows if r["workspace"] in workspaces]

    def search(self, query, mode, top_k, workspaces, filters):
        self.calls.append({"query": query, "mode": mode, "top_k": top_k, "workspaces": list(workspaces), "filters": filters})
        result = {
            "rank": 1, "workspace": self.golden_root, "file_id": f"{GOLDEN_FILE}.wav", "speaker": "Cass Sunstein",
            "speaker_label": "SPEAKER_01", "start_seconds": 64.0, "end_seconds": 71.5,
            "text": "the constitution <b>matters</b>", "highlight": "the <mark>constitution</mark> &lt;b&gt;matters&lt;/b&gt;",
            "matched_by": "words", **self.result,
        }
        return SearchResponse(results=[result] if query != "nothing" else [], timings_ms={"total": 42.0})


class SearchTestCase(AppTestCase):

    def setUp(self):
        super().setUp()
        st.cache_resource.clear()
        self.engine = FakeEngine(self.golden.root, self.uploads.root)
        self.patch = patch("src.search.engine.SearchEngine", lambda: self.engine)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        st.cache_resource.clear()
        super().tearDown()

    def search(self, query="constitution", **session):
        at = self.app(view="🔍 Search", **session).run()
        at.text_input(key="query").input(query).run()
        return at


class TestSearchView(SearchTestCase):

    def test_search_is_the_default_view(self):
        at = AppTest.from_file(APP, default_timeout=60).run()
        self.assertFalse(at.exception)
        self.assertIn("Search conversations", [h.value for h in at.header])
        self.assertEqual(self.engine.calls, [])

    def test_result_shows_file_speaker_timestamps_highlight_and_clip(self):
        at = self.search()
        self.assertFalse(at.exception)
        heading = " ".join(m.value for m in at.markdown)
        self.assertIn(f"**1. {GOLDEN_FILE}** · Cass Sunstein · `1:04–1:11`", heading)
        html = " ".join(h.proto.body for h in at.get("html"))
        self.assertIn("<mark>constitution</mark>", html)
        self.assertNotIn("<b>", html)
        self.assertEqual(len(at.get("audio")), 1)
        self.assertIn("1 results · 42 ms", [c.value for c in at.caption])

    def test_defaults_search_golden_set_hybrid_top_5_without_filters(self):
        self.search()
        self.assertEqual(self.engine.calls, [{"query": "constitution", "mode": "hybrid", "top_k": 5,
                                              "workspaces": [self.golden.root], "filters": SearchFilters()}])

    def test_mode_top_k_and_workspaces_are_passed_through(self):
        at = self.app(view="🔍 Search").run()
        at.radio(key="search_mode").set_value("Keyword")
        at.checkbox(key="search_uploads").check()
        at.number_input(key="top_k").set_value(8)
        at.text_input(key="query").input("constitution").run()
        call = self.engine.calls[-1]
        self.assertEqual((call["mode"], call["top_k"], call["workspaces"]), ("lexical", 8, [self.golden.root, self.uploads.root]))

    def test_speaker_filter_by_name_spans_recordings(self):
        at = self.app(view="🔍 Search").run()
        self.assertEqual(at.multiselect(key="filter_speakers").options, ["Cass Sunstein", "Tyler Cowen"])
        at.multiselect(key="filter_speakers").select("Tyler Cowen")
        at.text_input(key="query").input("constitution").run()
        self.assertEqual(self.engine.calls[-1]["filters"], SearchFilters(speakers=((1, "SPEAKER_00"), (3, "SPEAKER_01"))))

    def test_recording_filter_and_unnamed_upload_speakers(self):
        at = self.app(view="🔍 Search", search_uploads=True).run()
        self.assertIn("SPEAKER_00 (upload_x)", at.multiselect(key="filter_speakers").options)
        self.assertEqual(at.multiselect(key="filter_recordings").options, [GOLDEN_FILE, "upload_x", "other"])
        at.multiselect(key="filter_recordings").select(2)
        at.text_input(key="query").input("constitution").run()
        self.assertEqual(self.engine.calls[-1]["filters"], SearchFilters(file_pks=(2,)))

    def test_no_workspace_selected_asks_for_one(self):
        at = self.app(view="🔍 Search", search_golden=False).run()
        self.assertTrue(any("at least one place" in i.value for i in at.info))

    def test_no_matches_message(self):
        at = self.search("nothing")
        self.assertTrue(any("No matches" in i.value for i in at.info))

    def test_database_down_shows_how_to_start_it(self):
        self.engine.fail = True
        at = self.search()
        self.assertFalse(at.exception)
        self.assertTrue(any("docker compose up -d db" in e.value for e in at.error))
        self.assertEqual(self.engine.calls, [])


class TestSearchResultCard(SearchTestCase):
    """Fixture turns: #4 SPEAKER_01 0:48–1:00, #5 SPEAKER_00 1:01–1:45 (holds the 1:04–1:11 clip), #6 SPEAKER_01 1:45–1:53."""

    def html(self, at):
        return " ".join(h.proto.body for h in at.get("html"))

    def conversation(self, **result):
        self.engine.result = result
        at = self.search()
        at.toggle(key=self.toggle_key(at)).set_value(True).run()
        self.assertFalse(at.exception)
        return at

    def toggle_key(self, at):
        return next(t.key for t in at.toggle if t.key.startswith("conv_"))

    def button(self, at, prefix):
        return next(b for b in at.button if b.key and b.key.startswith(prefix))

    def turn_times(self, at):
        return re.findall(r"<code>(\d+:\d\d)</code>", self.html(at))

    def test_matched_by_label_for_each_kind(self):
        for kind, label in (("words", "🔤 words"), ("meaning", "💡 meaning"), ("both", "🔤💡 words + meaning")):
            self.engine.result = {"matched_by": kind}
            at = self.search()
            heading = " ".join(m.value for m in at.markdown)
            self.assertIn(f"`1:04–1:11` · :gray[matched by {label}]", heading, kind)

    def test_meaning_sentence_is_styled_and_escaped(self):
        self.engine.result = {"matched_by": "meaning", "highlight": '<span class="meaning">why &lt;liberalism&gt; lasts</span>'}
        at = self.search()
        body = self.html(at)
        self.assertIn('<span class="meaning">why &lt;liberalism&gt; lasts</span>', body)
        self.assertIn(".meaning {", body)

    def test_conversation_is_hidden_until_toggled(self):
        at = self.search()
        self.assertEqual(self.turn_times(at), [])
        self.assertEqual(len(at.get("audio")), 1)

    def test_conversation_shows_previous_matched_and_next_turn_with_clip_shaded(self):
        at = self.conversation()
        self.assertEqual(self.turn_times(at), ["0:48", "1:01", "1:45"])
        body = self.html(at)
        self.assertEqual(body.count('class="turn matched"'), 1)
        self.assertIn('<span class="clip">', body)
        shaded = body.split('<span class="clip">', 1)[1].split("</span>", 1)[0]
        self.assertTrue(shaded.strip())
        self.assertLess(len(shaded), len(body.split('class="turn matched"', 1)[1]))

    def test_step_earlier_and_later_moves_the_window_one_turn(self):
        at = self.conversation()
        self.button(at, "later_").click().run()
        self.assertEqual(self.turn_times(at), ["1:01", "1:45", "1:54"])
        self.button(at, "earlier_").click().run()
        self.button(at, "earlier_").click().run()
        self.assertEqual(self.turn_times(at), ["0:46", "0:48", "1:01"])

    def test_steps_stop_at_the_start_and_end_of_the_recording(self):
        at = self.conversation(start_seconds=2.0, end_seconds=6.0, speaker_label="SPEAKER_01")
        self.assertEqual(self.turn_times(at), ["0:00", "0:39"])
        self.assertTrue(self.button(at, "earlier_").disabled)
        self.assertFalse(self.button(at, "later_").disabled)

        with open(self.golden.canonical_path(GOLDEN_FILE), encoding="utf-8") as f:
            last = json.load(f)["turns"][-1]
        at = self.conversation(start_seconds=last["start_seconds"], end_seconds=last["end_seconds"], speaker_label=last["speaker_label"])
        self.assertEqual(len(self.turn_times(at)), 2)
        self.assertTrue(self.button(at, "later_").disabled)

    def test_play_exchange_adds_a_player_for_the_three_turns(self):
        at = self.conversation()
        self.assertEqual(len(at.get("audio")), 1)
        play = self.button(at, "playbtn_")
        self.assertIn("0:48–1:53", play.label)
        play.click().run()
        self.assertEqual(len(at.get("audio")), 2)
        self.button(at, "later_").click().run()
        self.assertEqual(len(at.get("audio")), 1)

    def test_open_transcript_jumps_to_the_recording_at_the_clip_time(self):
        at = self.conversation()
        self.button(at, "jump_").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state["view"], "🎙️ Recordings")
        self.assertEqual(at.session_state["workspace_name"], "Golden set")
        self.assertEqual(at.session_state["selected"], f"{GOLDEN_FILE}.wav")
        self.assertTrue(any("Opened from search at `1:04`" in m.value for m in at.markdown))
        focused = [m.value for m in at.markdown if m.value.startswith("👉")]
        self.assertEqual(len(focused), 1)
        self.assertIn("`1:01`", focused[0])
        self.assertTrue(any(a.proto.start_time == 64 for a in at.get("audio")))

    def test_picking_another_recording_clears_the_jump(self):
        at = self.conversation()
        self.button(at, "jump_").click().run()
        at.button(key=f"pick_Golden set_{GOLDEN_FILE}.wav").click().run()
        self.assertFalse(any(m.value.startswith("👉") for m in at.markdown))

    def test_query_mode_places_and_filters_survive_a_visit_to_recordings(self):
        at = self.app(view="🔍 Search", search_uploads=True).run()
        at.radio(key="search_mode").set_value("Keyword")
        at.number_input(key="top_k").set_value(8)
        at.multiselect(key="filter_speakers").select("Tyler Cowen")
        at.text_input(key="query").input("constitution").run()
        first = self.engine.calls[-1]

        at.radio(key="view").set_value("🎙️ Recordings").run()
        self.assertNotIn("Search conversations", [h.value for h in at.header])
        at.radio(key="view").set_value("🔍 Search").run()

        self.assertFalse(at.exception)
        self.assertEqual(at.text_input(key="query").value, "constitution")
        self.assertEqual(at.radio(key="search_mode").value, "Keyword")
        self.assertTrue(at.checkbox(key="search_uploads").value)
        self.assertEqual(at.number_input(key="top_k").value, 8)
        self.assertEqual(at.multiselect(key="filter_speakers").value, ["Tyler Cowen"])
        self.assertEqual(self.engine.calls[-1], first)

    def test_filters_no_longer_offered_are_dropped(self):
        at = self.app(view="🔍 Search", search_uploads=True).run()
        at.multiselect(key="filter_recordings").select(2).run()
        at.checkbox(key="search_uploads").uncheck()
        at.text_input(key="query").input("constitution").run()
        self.assertFalse(at.exception)
        self.assertEqual(at.multiselect(key="filter_recordings").value, [])
        self.assertEqual(self.engine.calls[-1]["filters"], SearchFilters())


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
        with patch("src.db.connection.connect"), patch("src.search.indexer.Indexer"):
            at = self.save_names()

        doc = load_labels(self.golden, f"{GOLDEN_FILE}.wav")
        self.assertEqual(doc["labels"], {"SPEAKER_00": "Tyler Cowen", "SPEAKER_01": "Cass Sunstein"})
        self.assertIn("Speaker names saved.", [m.value for m in at.success])
        self.assertEqual(doc["labeled_by"], "golden_simulated")
        self.assertFalse(os.path.exists(os.path.join(GOLDEN.labels_dir, f"{GOLDEN_FILE}.json.tmp")))

    def save_names(self):
        at = self.open_golden()
        at.text_input(key=self.name_key("SPEAKER_00")).input("Tyler Cowen")
        at.text_input(key=self.name_key("SPEAKER_01")).input("Cass Sunstein")
        next(b for b in at.button if b.label == "Save names").click().run()
        return at

    def test_saving_names_syncs_them_to_search(self):
        conn, indexer = MagicMock(), MagicMock()
        with patch("src.db.connection.connect", return_value=conn), patch("src.search.indexer.Indexer", return_value=indexer) as cls:
            self.save_names()
        self.assertEqual(cls.call_args.args[1].root, self.golden.root)
        indexer.sync_speaker_names.assert_called_once_with(f"{GOLDEN_FILE}.wav")
        conn.close.assert_called_once()

    def test_names_still_save_when_search_database_is_down(self):
        with patch("src.db.connection.connect", side_effect=psycopg2.OperationalError("down")):
            at = self.save_names()
        self.assertIsNotNone(load_labels(self.golden, f"{GOLDEN_FILE}.wav"))
        self.assertTrue(any("could not be updated" in w.value for w in at.warning))

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
