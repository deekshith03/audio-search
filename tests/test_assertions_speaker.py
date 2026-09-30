import json
import os
import tempfile
import unittest

from evals import assertions


class TestResolveResultSpeaker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with open(os.path.join(self.tmp.name, "audio_01_x.json"), "w") as f:
            json.dump({"labels": {"SPEAKER_00": "Lex Fridman", "SPEAKER_01": "DHH"}}, f)
        assertions._speaker_names.cache_clear()

    def tearDown(self):
        self.tmp.cleanup()
        assertions._speaker_names.cache_clear()

    def resolve(self, result):
        return assertions.resolve_result_speaker(result, labels_dir=self.tmp.name)

    def test_anonymous_label_is_resolved_to_name(self):
        self.assertEqual(self.resolve({"file_id": "audio_01_x.wav", "speaker": "SPEAKER_01"}), "DHH")

    def test_speaker_label_field_is_used_when_speaker_missing(self):
        self.assertEqual(self.resolve({"file_id": "audio_01_x.wav", "speaker_label": "SPEAKER_00"}), "Lex Fridman")

    def test_real_names_pass_through(self):
        self.assertEqual(self.resolve({"file_id": "audio_01_x.wav", "speaker": "DHH"}), "DHH")

    def test_unlabeled_file_keeps_anonymous_label(self):
        self.assertEqual(self.resolve({"file_id": "audio_02_y.wav", "speaker": "SPEAKER_00"}), "SPEAKER_00")

    def test_missing_speaker_is_none(self):
        self.assertIsNone(self.resolve({"file_id": "audio_01_x.wav"}))


class TestAssertionUsesResolvedSpeaker(unittest.TestCase):

    def test_anonymous_result_matches_named_qrel_after_labeling(self):
        context = {"vars": {
            "query_id": "T-1",
            "category": "single_file",
            "expected_moments": [{"file_id": "audio_01_x.wav", "speaker": "DHH", "start_seconds": 10.0, "end_seconds": 20.0}],
            "hard_negatives": [],
        }}
        output = {"results": [{"file_id": "audio_01_x.wav", "speaker": "SPEAKER_01", "start_seconds": 10.0, "end_seconds": 20.0}]}
        original = assertions.resolve_result_speaker
        try:
            assertions.resolve_result_speaker = lambda res: {"SPEAKER_01": "DHH"}.get(res["speaker"], res["speaker"])
            self.assertTrue(assertions.get_assert(output, context)["pass"])
            output["results"][0]["speaker"] = "SPEAKER_00"
            self.assertFalse(assertions.get_assert(output, context)["pass"])
        finally:
            assertions.resolve_result_speaker = original



class TestAssertionAlternatives(unittest.TestCase):

    def context(self):
        return {"vars": {
            "query_id": "T-2", "category": "single_file", "hard_negatives": [],
            "expected_moments": [{
                "file_id": "a.wav", "speaker": "Gary", "start_seconds": 10.0, "end_seconds": 14.0,
                "alternatives": [{"file_id": "b.wav", "speaker": "Gary", "start_seconds": 50.0, "end_seconds": 54.0}],
            }],
        }}

    def result(self, file_id, start):
        return {"results": [{"file_id": file_id, "speaker": "Gary", "start_seconds": start, "end_seconds": start + 4.0}]}

    def test_result_on_an_alternative_passes(self):
        self.assertTrue(assertions.get_assert(self.result("b.wav", 50.0), self.context())["pass"])
        self.assertEqual(assertions.get_assert(self.result("b.wav", 50.0), self.context())["score"], 1.0)

    def test_result_on_the_moment_passes_and_elsewhere_fails(self):
        self.assertTrue(assertions.get_assert(self.result("a.wav", 10.0), self.context())["pass"])
        self.assertFalse(assertions.get_assert(self.result("b.wav", 200.0), self.context())["pass"])


if __name__ == "__main__":
    unittest.main()
