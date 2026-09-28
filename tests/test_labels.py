import os
import tempfile
import unittest

from src.pipeline.common import Workspace
from src.pipeline.labels import (
    LabelValidationError,
    labels_path,
    load_labels,
    save_labels,
    speaker_name,
    swap_labels,
    validate_labels,
)

SPEAKERS = ["SPEAKER_00", "SPEAKER_01"]


class TestValidateLabels(unittest.TestCase):

    def test_trims_and_collapses_whitespace(self):
        self.assertEqual(
            validate_labels({"SPEAKER_00": "  Lex   Fridman ", "SPEAKER_01": "DHH"}, SPEAKERS),
            {"SPEAKER_00": "Lex Fridman", "SPEAKER_01": "DHH"},
        )

    def test_rejects_empty_name(self):
        with self.assertRaisesRegex(LabelValidationError, "SPEAKER_01 is empty"):
            validate_labels({"SPEAKER_00": "Lex", "SPEAKER_01": "   "}, SPEAKERS)

    def test_rejects_duplicate_names_case_insensitively(self):
        with self.assertRaisesRegex(LabelValidationError, "different name"):
            validate_labels({"SPEAKER_00": "Lex", "SPEAKER_01": "lex"}, SPEAKERS)

    def test_rejects_missing_or_extra_speakers(self):
        with self.assertRaises(LabelValidationError):
            validate_labels({"SPEAKER_00": "Lex"}, SPEAKERS)
        with self.assertRaises(LabelValidationError):
            validate_labels({"SPEAKER_00": "Lex", "SPEAKER_01": "DHH", "SPEAKER_02": "X"}, SPEAKERS)

    def test_rejects_overlong_name(self):
        with self.assertRaisesRegex(LabelValidationError, "longer than"):
            validate_labels({"SPEAKER_00": "x" * 81, "SPEAKER_01": "DHH"}, SPEAKERS)


class TestSaveAndLoad(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = Workspace(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip(self):
        doc = save_labels(self.ws, "clip.wav", {"SPEAKER_00": "Lex", "SPEAKER_01": "DHH"}, SPEAKERS)
        self.assertEqual(labels_path(self.ws, "clip.wav"), os.path.join(self.tmp.name, "speaker_labels", "clip.json"))
        self.assertEqual(load_labels(self.ws, "clip.wav"), doc)
        self.assertEqual(doc["labeled_by"], "user")
        self.assertTrue(doc["labeled_at"])

    def test_relabel_overwrites(self):
        save_labels(self.ws, "clip.wav", {"SPEAKER_00": "Lex", "SPEAKER_01": "DHH"}, SPEAKERS)
        save_labels(self.ws, "clip.wav", {"SPEAKER_00": "DHH", "SPEAKER_01": "Lex"}, SPEAKERS)
        self.assertEqual(load_labels(self.ws, "clip.wav")["labels"]["SPEAKER_00"], "DHH")

    def test_golden_simulated_provenance(self):
        doc = save_labels(self.ws, "clip.wav", {"SPEAKER_00": "A", "SPEAKER_01": "B"}, SPEAKERS, labeled_by="golden_simulated")
        self.assertEqual(doc["labeled_by"], "golden_simulated")

    def test_rejects_unknown_provenance(self):
        with self.assertRaises(LabelValidationError):
            save_labels(self.ws, "clip.wav", {"SPEAKER_00": "A", "SPEAKER_01": "B"}, SPEAKERS, labeled_by="oracle")

    def test_invalid_labels_are_not_written(self):
        with self.assertRaises(LabelValidationError):
            save_labels(self.ws, "clip.wav", {"SPEAKER_00": "A", "SPEAKER_01": "a"}, SPEAKERS)
        self.assertIsNone(load_labels(self.ws, "clip.wav"))

    def test_missing_labels_load_as_none(self):
        self.assertIsNone(load_labels(self.ws, "nothing.wav"))


class TestHelpers(unittest.TestCase):

    def test_swap(self):
        self.assertEqual(swap_labels({"SPEAKER_00": "A", "SPEAKER_01": "B"}), {"SPEAKER_00": "B", "SPEAKER_01": "A"})

    def test_swap_requires_two_speakers(self):
        with self.assertRaises(LabelValidationError):
            swap_labels({"SPEAKER_00": "A"})

    def test_speaker_name_falls_back_to_label(self):
        doc = {"labels": {"SPEAKER_00": "Lex"}}
        self.assertEqual(speaker_name(doc, "SPEAKER_00"), "Lex")
        self.assertEqual(speaker_name(doc, "SPEAKER_01"), "SPEAKER_01")
        self.assertEqual(speaker_name(None, "SPEAKER_00"), "SPEAKER_00")


if __name__ == "__main__":
    unittest.main()
