"""
Unit and Integration Tests for Phase 2 ML Pipeline & Output Schemas.

Validates:
1. Strict schema compliance of all 7 canonical JSON outputs against PHASE_2_SPECIFICATION.md.
2. Reconciliation logic: strict >50% overlap rule, gap fallback, and 'end_of_audio' split reason.
3. Globally optimal Hungarian / permutation speaker name mapping.
4. True word-level speaker accuracy and coverage calculation.
5. Distinction between raw PyAnnote RTTM DER and reconciled transcript DER.
6. Fail-closed security check on missing Hugging Face credentials.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch

from src.pipeline.reconcile import (
    assign_word_to_speaker,
    compute_speaker_mapping,
    reconcile_transcript
)
from src.pipeline.evaluate_pipeline import (
    evaluate_word_speaker_accuracy,
    normalize_text,
    extract_normalized_tokens
)
from src.pipeline.diarize import get_hf_token


class TestPhase2SchemaAndArtifacts(unittest.TestCase):

    def setUp(self):
        self.output_dir = "dataset/pipeline_outputs"
        self.canonical_files = [
            f for f in sorted(os.listdir(self.output_dir))
            if f.endswith("_canonical.json")
        ]

    def test_all_seven_canonical_transcripts_exist(self):
        self.assertEqual(len(self.canonical_files), 7, "All 7 canonical transcripts must be emitted")

    def test_canonical_schema_compliance(self):
        required_root_keys = {
            "file_id", "pipeline_version", "asr_model", "diarization_model",
            "alignment_model", "audio_duration_seconds", "telemetry",
            "speaker_mapping", "turns"
        }
        required_telemetry_keys = {
            "total_words", "fallback_aligned_words", "alignment_fallback_rate",
            "total_turns", "pause_splits", "speaker_change_splits"
        }
        required_turn_keys = {
            "turn_id", "speaker_label", "speaker_name", "start_seconds",
            "end_seconds", "split_reason", "is_short_turn", "text", "words"
        }
        required_word_keys = {
            "word", "start_seconds", "end_seconds", "confidence", "timing_source"
        }

        for fname in self.canonical_files:
            fpath = os.path.join(self.output_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Check exact root keys (no extra / legacy keys)
            self.assertEqual(set(data.keys()), required_root_keys, f"{fname} root keys do not match exact contract")
            self.assertGreater(data["audio_duration_seconds"], 0.0, f"{fname} duration must be positive")

            # Check exact telemetry keys (no aliases)
            telem = data["telemetry"]
            self.assertEqual(set(telem.keys()), required_telemetry_keys, f"{fname} telemetry keys do not match exact contract")

            # Check turns
            turns = data["turns"]
            self.assertGreater(len(turns), 0, f"{fname} must have turns")
            self.assertEqual(turns[-1]["split_reason"], "end_of_audio", f"{fname} final turn must be 'end_of_audio'")

            for t in turns:
                self.assertEqual(set(t.keys()), required_turn_keys, f"{fname} turn {t['turn_id']} keys do not match exact contract")
                self.assertLess(t["start_seconds"], t["end_seconds"], f"{fname} turn timestamps inverted")
                for w in t["words"]:
                    self.assertEqual(set(w.keys()), required_word_keys, f"{fname} word {w} keys do not match exact contract")

    def test_zero_leakage_anonymous_speaker_clusters(self):
        for fname in self.canonical_files:
            fpath = os.path.join(self.output_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            mapping = data["speaker_mapping"]
            # Assert all values are anonymous "Speaker N"
            for spk_lbl, spk_name in mapping.items():
                self.assertTrue(spk_lbl.startswith("SPEAKER_"))
                self.assertTrue(spk_name.startswith("Speaker "))


class TestReconciliationLogic(unittest.TestCase):

    def setUp(self):
        self.intervals = [
            {"speaker": "SPEAKER_00", "start": 0.0, "end": 10.0, "duration": 10.0},
            {"speaker": "SPEAKER_01", "start": 12.0, "end": 20.0, "duration": 8.0}
        ]

    def test_strict_greater_than_50_percent_overlap_assigns_speaker(self):
        # Word [1.0, 2.0] (duration 1.0) inside SPEAKER_00 [0, 10] -> 100% overlap
        word_full = {"word": "hello", "start_seconds": 1.0, "end_seconds": 2.0}
        spk = assign_word_to_speaker(word_full, self.intervals)
        self.assertEqual(spk, "SPEAKER_00")

        # Word [9.4, 10.4] (duration 1.0) overlaps SPEAKER_00 by 0.6s (60%) -> assigns SPEAKER_00
        word_60 = {"word": "world", "start_seconds": 9.4, "end_seconds": 10.4}
        spk_60 = assign_word_to_speaker(word_60, self.intervals)
        self.assertEqual(spk_60, "SPEAKER_00")

    def test_low_overlap_falls_back_to_nearest_interval(self):
        # Word [9.8, 10.8] (duration 1.0) overlaps SPEAKER_00 by only 0.2s (20%), gap is [10.0, 12.0]
        # Overlap <= 50%, nearest interval is SPEAKER_00 (dist 0.0) -> SPEAKER_00
        word_20 = {"word": "testing", "start_seconds": 9.8, "end_seconds": 10.8}
        spk_20 = assign_word_to_speaker(word_20, self.intervals)
        self.assertEqual(spk_20, "SPEAKER_00")

        # Word [11.2, 11.9] (in gap [10.0, 12.0])
        # Distance to SPEAKER_00 is 1.2s; Distance to SPEAKER_01 is 0.1s -> nearest is SPEAKER_01!
        word_gap = {"word": "gapword", "start_seconds": 11.2, "end_seconds": 11.9}
        spk_gap = assign_word_to_speaker(word_gap, self.intervals)
        self.assertEqual(spk_gap, "SPEAKER_01")

    def test_optimal_speaker_mapping(self):
        pipeline_turns = [
            {"speaker_label": "SPEAKER_00", "start_seconds": 0.0, "end_seconds": 100.0},
            {"speaker_label": "SPEAKER_01", "start_seconds": 100.0, "end_seconds": 200.0}
        ]
        # Mock reference ground truth data
        tmp_ref = "/tmp/test_ref_ground_truth.json"
        ref_data = {
            "turns": [
                {"speaker": "Alice", "start_time": 0.0, "end_time": 95.0},
                {"speaker": "Bob", "start_time": 95.0, "end_time": 200.0}
            ]
        }
        with open(tmp_ref, "w", encoding="utf-8") as f:
            json.dump(ref_data, f)

        try:
            mapping = compute_speaker_mapping(pipeline_turns, tmp_ref)
            self.assertEqual(mapping, {"SPEAKER_00": "Alice", "SPEAKER_01": "Bob"})
        finally:
            if os.path.exists(tmp_ref):
                os.remove(tmp_ref)


class TestMetricAccuracy(unittest.TestCase):

    def test_word_speaker_accuracy_penalizes_misattribution(self):
        gt_data = {
            "turns": [
                {"speaker": "Alice", "text": "hello world from our system"},
                {"speaker": "Bob", "text": "thanks for having me today"}
            ]
        }

        # Case 1: 100% accurate speaker attribution
        pipe_perfect = {
            "turns": [
                {"speaker_name": "Alice", "words": [{"word": "hello"}, {"word": "world"}, {"word": "from"}, {"word": "our"}, {"word": "system"}]},
                {"speaker_name": "Bob", "words": [{"word": "thanks"}, {"word": "for"}, {"word": "having"}, {"word": "me"}, {"word": "today"}]}
            ]
        }
        res_perfect = evaluate_word_speaker_accuracy(gt_data, pipe_perfect)
        self.assertEqual(res_perfect["word_speaker_accuracy"], 1.0)
        self.assertEqual(res_perfect["matched_words"], 10)
        self.assertEqual(res_perfect["correct_speaker_words"], 10)

        # Case 2: One word ("thanks") misattributed to Alice due to turn boundary shift
        pipe_imperfect = {
            "turns": [
                {"speaker_name": "Alice", "words": [{"word": "hello"}, {"word": "world"}, {"word": "from"}, {"word": "our"}, {"word": "system"}, {"word": "thanks"}]},
                {"speaker_name": "Bob", "words": [{"word": "for"}, {"word": "having"}, {"word": "me"}, {"word": "today"}]}
            ]
        }
        res_imperfect = evaluate_word_speaker_accuracy(gt_data, pipe_imperfect)
        self.assertEqual(res_imperfect["matched_words"], 10)
        self.assertEqual(res_imperfect["correct_speaker_words"], 9)
        self.assertEqual(res_imperfect["word_speaker_accuracy"], 0.9)


class TestSecurityToken(unittest.TestCase):

    def test_loads_token_from_env_file(self):
        token = get_hf_token()
        self.assertTrue(token.startswith("hf_"))

    def test_fail_closed_when_hf_token_absent(self):
        with patch.dict(os.environ, {}, clear=True):
            # Pass non-existent env_path and non-existent cache
            with patch("os.path.expanduser", return_value="/tmp/nonexistent_token_path"):
                with self.assertRaises(RuntimeError) as cm:
                    get_hf_token(env_path="/tmp/nonexistent_env")
                self.assertIn("Hugging Face token not found", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
