"""
Phase 2 pipeline tests: canonical artifact contract, reconciliation logic, and evaluation metrics.
"""

import json
import os
import tempfile
import unittest

from src.pipeline.evaluate_pipeline import (
    compute_evaluation_speaker_mapping,
    evaluate_word_speaker_accuracy,
    extract_normalized_tokens,
    normalize_text,
)
from src.pipeline.reconcile import (
    assign_word_to_speaker,
    group_into_turns,
    reconcile_transcript,
    smooth_isolated_words,
)

OUTPUT_DIR = "dataset/pipeline_outputs"


def word(text, start, end, confidence=0.9, source="wav2vec2_aligned"):
    return {"word": text, "start_seconds": start, "end_seconds": end, "confidence": confidence, "timing_source": source}


class TestCanonicalArtifacts(unittest.TestCase):

    ROOT_KEYS = {
        "file_id", "pipeline_version", "asr_model", "alignment_model", "diarization_model",
        "audio_duration_seconds", "upstream_cache_keys", "telemetry", "speaker_labels", "turns",
    }
    TELEMETRY_KEYS = {
        "total_words", "fallback_aligned_words", "alignment_fallback_rate", "wildcard_aligned_words",
        "total_turns", "pause_splits", "speaker_change_splits",
    }
    TURN_KEYS = {"turn_id", "speaker_label", "start_seconds", "end_seconds", "split_reason", "is_short_turn", "text", "words"}
    WORD_KEYS = {"word", "start_seconds", "end_seconds", "confidence", "timing_source"}

    @classmethod
    def setUpClass(cls):
        cls.docs = {}
        for fname in sorted(os.listdir(OUTPUT_DIR)):
            if fname.endswith("_canonical.json"):
                with open(os.path.join(OUTPUT_DIR, fname), encoding="utf-8") as f:
                    cls.docs[fname] = json.load(f)

    def test_all_seven_canonical_transcripts_exist(self):
        self.assertEqual(len(self.docs), 7)

    def test_root_telemetry_turn_and_word_keys_match_contract(self):
        for fname, doc in self.docs.items():
            self.assertEqual(set(doc), self.ROOT_KEYS, fname)
            self.assertEqual(set(doc["telemetry"]), self.TELEMETRY_KEYS, fname)
            for t in doc["turns"]:
                self.assertEqual(set(t), self.TURN_KEYS, f"{fname} turn {t['turn_id']}")
                for w in t["words"]:
                    self.assertEqual(set(w), self.WORD_KEYS, f"{fname} word {w}")

    def test_models_reflect_single_cpu_backend(self):
        for fname, doc in self.docs.items():
            self.assertEqual(doc["asr_model"], "mobiuslabsgmbh/faster-whisper-large-v3-turbo", fname)
            self.assertEqual(doc["alignment_model"], "WAV2VEC2_ASR_BASE_960H", fname)
            self.assertEqual(doc["diarization_model"], "pyannote/speaker-diarization-community-1", fname)

    def test_exactly_two_anonymous_speakers_and_no_names(self):
        for fname, doc in self.docs.items():
            self.assertEqual(doc["speaker_labels"], ["SPEAKER_00", "SPEAKER_01"], fname)
            self.assertEqual({t["speaker_label"] for t in doc["turns"]}, set(doc["speaker_labels"]), fname)

    def test_turns_are_ordered_positive_and_within_audio(self):
        for fname, doc in self.docs.items():
            turns = doc["turns"]
            self.assertEqual([t["turn_id"] for t in turns], list(range(1, len(turns) + 1)), fname)
            self.assertEqual(turns[-1]["split_reason"], "end_of_audio", fname)
            for prev, curr in zip(turns, turns[1:]):
                self.assertLessEqual(prev["start_seconds"], curr["start_seconds"], fname)
            for t in turns:
                self.assertLess(t["start_seconds"], t["end_seconds"], f"{fname} turn {t['turn_id']}")
                self.assertLessEqual(t["end_seconds"], doc["audio_duration_seconds"] + 0.05, fname)
                self.assertEqual(t["text"], " ".join(w["word"] for w in t["words"]), fname)

    def test_telemetry_is_consistent_with_turns(self):
        for fname, doc in self.docs.items():
            tel = doc["telemetry"]
            self.assertEqual(tel["total_turns"], len(doc["turns"]), fname)
            self.assertEqual(tel["total_words"], sum(len(t["words"]) for t in doc["turns"]), fname)
            self.assertEqual(tel["pause_splits"] + tel["speaker_change_splits"] + 1, len(doc["turns"]), fname)
            interpolated = sum(1 for t in doc["turns"] for w in t["words"] if w["timing_source"] == "interpolated_fallback")
            self.assertEqual(tel["fallback_aligned_words"], interpolated, fname)


class TestWordAssignment(unittest.TestCase):

    def setUp(self):
        self.intervals = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0},
            {"speaker": "SPEAKER_01", "start": 12.0, "end": 20.0},
        ]

    def test_majority_overlap_assigns_speaker(self):
        self.assertEqual(assign_word_to_speaker(word("hello", 1.0, 2.0), self.intervals), "SPEAKER_00")
        self.assertEqual(assign_word_to_speaker(word("world", 9.4, 10.4), self.intervals), "SPEAKER_00")

    def test_low_overlap_falls_back_to_nearest_interval(self):
        self.assertEqual(assign_word_to_speaker(word("testing", 9.8, 10.8), self.intervals), "SPEAKER_00")

    def test_gap_word_goes_to_nearest_interval(self):
        self.assertEqual(assign_word_to_speaker(word("gap", 11.2, 11.9), self.intervals), "SPEAKER_01")

    def test_overlap_is_summed_across_same_speaker_intervals(self):
        intervals = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.3},
            {"speaker": "SPEAKER_01", "start": 0.3, "end": 0.6},
            {"speaker": "SPEAKER_00", "start": 0.6, "end": 1.0},
        ]
        self.assertEqual(assign_word_to_speaker(word("x", 0.0, 1.0), intervals), "SPEAKER_00")


class TestSmoothing(unittest.TestCase):

    def seq(self, speakers, words):
        return list(zip(speakers, words))

    def test_isolated_word_with_tight_gaps_is_folded(self):
        words = [word("the", 0.0, 0.2), word("big", 0.25, 0.5), word("dog", 0.55, 0.8)]
        out = smooth_isolated_words(self.seq(["A", "B", "A"], words))
        self.assertEqual([s for s, _ in out], ["A", "A", "A"])

    def test_backchannel_word_is_kept(self):
        words = [word("so", 0.0, 0.2), word("yeah", 0.25, 0.5), word("then", 0.55, 0.8)]
        out = smooth_isolated_words(self.seq(["A", "B", "A"], words))
        self.assertEqual([s for s, _ in out], ["A", "B", "A"])

    def test_isolated_word_after_long_pause_is_kept(self):
        words = [word("so", 0.0, 0.2), word("well", 1.0, 1.2), word("then", 1.25, 1.5)]
        out = smooth_isolated_words(self.seq(["A", "B", "A"], words))
        self.assertEqual([s for s, _ in out], ["A", "B", "A"])

    def test_two_word_run_is_not_folded(self):
        words = [word("a", 0.0, 0.1), word("b", 0.15, 0.25), word("c", 0.3, 0.4), word("d", 0.45, 0.55)]
        out = smooth_isolated_words(self.seq(["A", "B", "B", "A"], words))
        self.assertEqual([s for s, _ in out], ["A", "B", "B", "A"])

    def test_decisions_read_original_assignments(self):
        words = [word(t, i * 0.3, i * 0.3 + 0.2) for i, t in enumerate("abcde")]
        out = smooth_isolated_words(self.seq(["B", "A", "B", "A", "B"], words))
        self.assertEqual([s for s, _ in out], ["B", "B", "A", "B", "B"])


class TestTurnGrouping(unittest.TestCase):

    def test_speaker_change_pause_and_end_of_audio(self):
        assignments = [
            ("SPEAKER_00", word("hello", 0.0, 0.5)),
            ("SPEAKER_00", word("there", 0.6, 1.0)),
            ("SPEAKER_01", word("hi", 1.2, 1.5)),
            ("SPEAKER_01", word("again", 3.5, 4.0)),
        ]
        turns, pauses, changes = group_into_turns(assignments, pause_threshold=1.5)
        self.assertEqual([t["split_reason"] for t in turns], ["speaker_change", "pause", "end_of_audio"])
        self.assertEqual((pauses, changes), (1, 1))
        self.assertEqual([t["text"] for t in turns], ["hello there", "hi", "again"])
        self.assertEqual((turns[0]["start_seconds"], turns[0]["end_seconds"]), (0.0, 1.0))

    def test_pause_exactly_at_threshold_does_not_split(self):
        assignments = [("SPEAKER_00", word("a", 0.0, 0.5)), ("SPEAKER_00", word("b", 2.0, 2.5))]
        turns, pauses, _ = group_into_turns(assignments, pause_threshold=1.5)
        self.assertEqual((len(turns), pauses), (1, 0))

    def test_short_backchannel_turn_is_flagged(self):
        assignments = [("SPEAKER_00", word("so", 0.0, 1.0)), ("SPEAKER_01", word("Yeah.", 1.1, 1.4)), ("SPEAKER_00", word("ok", 1.5, 2.5))]
        turns, _, _ = group_into_turns(assignments)
        self.assertEqual([t["is_short_turn"] for t in turns], [False, True, False])

    def test_long_backchannel_and_short_content_are_not_flagged(self):
        long_yeah, _, _ = group_into_turns([("SPEAKER_00", word("yeah", 0.0, 1.0))])
        short_content, _, _ = group_into_turns([("SPEAKER_00", word("pgvector", 0.0, 0.4))])
        self.assertFalse(long_yeah[0]["is_short_turn"])
        self.assertFalse(short_content[0]["is_short_turn"])

    def test_empty_input_gives_no_turns(self):
        self.assertEqual(group_into_turns([]), ([], 0, 0))


class TestReconcileTranscript(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.aligned = os.path.join(self.tmp.name, "clip_aligned.json")
        self.diar = os.path.join(self.tmp.name, "clip_diarization.json")
        self.out = os.path.join(self.tmp.name, "out", "clip_canonical.json")
        self.aligned_doc = {
            "asr_model": "asr-model",
            "alignment_model": "align-model",
            "cache_key": "align-key",
            "telemetry": {"total_words": 4, "fallback_aligned_words": 1, "alignment_fallback_rate": 0.25, "wildcard_aligned_words": 0},
            "segments": [
                {"words": [word("Hello", 0.0, 0.4), word("there.", 0.5, 0.9)]},
                {"words": [word("Hi", 1.2, 1.5), word("$5", 1.5, 1.8, confidence=None, source="interpolated_fallback")]},
            ],
        }
        self.diar_doc = {
            "diarizer_model": "diar-model",
            "cache_key": "diar-key",
            "intervals": [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0},
                {"speaker": "SPEAKER_01", "start": 1.1, "end": 2.0},
            ],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def run_reconcile(self):
        for path, doc in ((self.aligned, self.aligned_doc), (self.diar, self.diar_doc)):
            with open(path, "w") as f:
                json.dump(doc, f)
        return reconcile_transcript("clip.wav", self.aligned, self.diar, self.out)

    def test_end_to_end_output(self):
        doc = self.run_reconcile()
        with open(self.out) as f:
            self.assertEqual(json.load(f), doc)

        self.assertEqual(doc["speaker_labels"], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual([t["text"] for t in doc["turns"]], ["Hello there.", "Hi $5"])
        self.assertEqual([t["speaker_label"] for t in doc["turns"]], ["SPEAKER_00", "SPEAKER_01"])
        self.assertEqual(doc["audio_duration_seconds"], 1.8)
        self.assertEqual(doc["upstream_cache_keys"], {"align": "align-key", "diarize": "diar-key"})
        self.assertEqual((doc["asr_model"], doc["alignment_model"], doc["diarization_model"]), ("asr-model", "align-model", "diar-model"))
        self.assertIsNone(doc["turns"][1]["words"][1]["confidence"])
        self.assertNotIn("speaker_name", doc["turns"][0])

    def test_alignment_telemetry_is_carried_through(self):
        tel = self.run_reconcile()["telemetry"]
        self.assertEqual(tel["fallback_aligned_words"], 1)
        self.assertEqual(tel["alignment_fallback_rate"], 0.25)
        self.assertEqual((tel["total_turns"], tel["speaker_change_splits"], tel["pause_splits"]), (2, 1, 0))

    def test_raises_without_intervals(self):
        self.diar_doc["intervals"] = []
        with self.assertRaises(ValueError):
            self.run_reconcile()

    def test_raises_without_words(self):
        self.aligned_doc["segments"] = [{"words": []}]
        with self.assertRaises(ValueError):
            self.run_reconcile()


class TestEvaluationMetrics(unittest.TestCase):

    def setUp(self):
        self.gt = {"turns": [
            {"speaker": "Alice", "start_time": 0.0, "end_time": 5.0, "text": "hello world from our system"},
            {"speaker": "Bob", "start_time": 5.0, "end_time": 10.0, "text": "thanks for having me today"},
        ]}

    def pipe(self, split_after):
        tokens = "hello world from our system thanks for having me today".split()
        return {"turns": [
            {"speaker_label": "SPEAKER_01", "start_seconds": 0.0, "end_seconds": 5.0, "words": [{"word": t} for t in tokens[:split_after]]},
            {"speaker_label": "SPEAKER_00", "start_seconds": 5.0, "end_seconds": 10.0, "words": [{"word": t} for t in tokens[split_after:]]},
        ]}

    def test_mapping_is_by_overlap_not_label_order(self):
        self.assertEqual(compute_evaluation_speaker_mapping(self.gt, self.pipe(5)), {"SPEAKER_01": "Alice", "SPEAKER_00": "Bob"})

    def test_perfect_attribution(self):
        res = evaluate_word_speaker_accuracy(self.gt, self.pipe(5))
        self.assertEqual((res["word_speaker_accuracy"], res["matched_words"], res["lexical_coverage"]), (1.0, 10, 1.0))

    def test_misattributed_word_is_penalized(self):
        res = evaluate_word_speaker_accuracy(self.gt, self.pipe(6))
        self.assertEqual((res["matched_words"], res["correct_speaker_words"], res["word_speaker_accuracy"]), (10, 9, 0.9))

    def test_hyphenated_hypothesis_words_contribute_every_token(self):
        gt = {"turns": [{"speaker": "Alice", "start_time": 0.0, "end_time": 2.0, "text": "state of the art"}]}
        pipe = {"turns": [{"speaker_label": "SPEAKER_00", "start_seconds": 0.0, "end_seconds": 2.0, "words": [{"word": "state-of-the-art"}]}]}
        self.assertEqual(evaluate_word_speaker_accuracy(gt, pipe)["matched_words"], 4)

    def test_normalizer_and_tokenizer(self):
        self.assertEqual(normalize_text("I’m using PG vector!"), "i am using pgvector")
        self.assertEqual(extract_normalized_tokens("Don't stop, OK?"), ["don't", "stop", "ok"])


if __name__ == "__main__":
    unittest.main()
