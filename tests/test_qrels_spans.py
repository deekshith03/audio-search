import json
import os
import unittest

from scripts.tighten_qrels import word_char_spans

from evals.qrels import SPLIT_PATHS

GT_DIR = "dataset/ground_truth"


class TestWordCharSpans(unittest.TestCase):

    def test_spans_map_back_to_words(self):
        text = "Hello  big world."
        self.assertEqual([text[s:e] for s, e in word_char_spans(text, ("Hello", "big", "world."))], ["Hello", "big", "world."])

    def test_token_split_by_aligner_is_located(self):
        text = 'not weakness."But if'
        spans = word_char_spans(text, ("not", 'weakness."', "But", "if"))
        self.assertEqual([text[s:e] for s, e in spans], ["not", 'weakness."', "But", "if"])

    def test_punctuation_shared_by_neighbours_is_tolerated(self):
        text = 'not weakness."But if'
        spans = word_char_spans(text, ("not", 'weakness."', '"But', "if"))
        self.assertEqual([text[s:e] for s, e in spans], ["not", 'weakness."', "But", "if"])

    def test_missing_word_raises(self):
        with self.assertRaises(RuntimeError):
            word_char_spans("hello world", ("goodbye",))


class TestTightenedQrels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.queries = []
        for path in SPLIT_PATHS.values():
            with open(path, encoding="utf-8") as f:
                cls.queries += json.load(f)["queries"]
        cls.turns = {}
        for name in os.listdir(GT_DIR):
            with open(os.path.join(GT_DIR, name), encoding="utf-8") as f:
                gt = json.load(f)
            cls.turns[gt["file_id"]] = {t["turn_id"]: t for t in gt["turns"]}

    def moments(self):
        for q in self.queries:
            for m in q["relevant_moments"]:
                yield q["query_id"], m

    def test_every_moment_keeps_original_turn_bounds(self):
        for qid, m in self.moments():
            turn = self.turns[m["file_id"]][m["turn_id"]]
            self.assertEqual((m["turn_start_seconds"], m["turn_end_seconds"]), (turn["start_time"], turn["end_time"]), qid)

    def test_span_is_inside_turn_and_positive(self):
        for qid, m in self.moments():
            self.assertLess(m["start_seconds"], m["end_seconds"], qid)
            self.assertGreaterEqual(m["start_seconds"], m["turn_start_seconds"] - 0.05, qid)
            self.assertLessEqual(m["end_seconds"], m["turn_end_seconds"] + 0.05, qid)

    def test_spans_are_narrower_than_long_turns(self):
        for qid, m in self.moments():
            turn_dur = m["turn_end_seconds"] - m["turn_start_seconds"]
            if turn_dur > 60:
                self.assertLess(m["end_seconds"] - m["start_seconds"], turn_dur / 2, qid)

    def test_span_source_is_reference_alignment(self):
        for qid, m in self.moments():
            self.assertTrue(m["span_source"].startswith("wav2vec2_forced_alignment_of_reference_turn"), qid)


if __name__ == "__main__":
    unittest.main()
