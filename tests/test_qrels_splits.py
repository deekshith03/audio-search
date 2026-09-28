import os
import unittest
from unittest.mock import patch

from evals import evaluate_recall
from evals.metrics import evaluate_retrieval, result_matches_moment
from evals.qrels import ANY_OF_CATEGORIES, CATEGORIES, EXPECTED_BREAKDOWN, SPLIT_PATHS, load_qrels
from evals.validate_dataset_integrity import validate_all


class TestQrelsModule(unittest.TestCase):

    def test_split_files_exist_and_declare_their_split(self):
        for split, path in SPLIT_PATHS.items():
            self.assertTrue(os.path.exists(path), path)
            self.assertEqual(load_qrels(split)["split"], split)

    def test_breakdown_matches_expected_and_is_60_40(self):
        for split in SPLIT_PATHS:
            queries = load_qrels(split)["queries"]
            counts = {c: sum(q["category"] == c for q in queries) for c in CATEGORIES}
            self.assertEqual(counts, EXPECTED_BREAKDOWN[split])
        n_test, n_dev = (sum(EXPECTED_BREAKDOWN[s].values()) for s in ("test", "dev"))
        self.assertAlmostEqual(n_test / (n_test + n_dev), 0.6, places=2)

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            load_qrels("train")

    def test_short_keyword_queries_are_short(self):
        for split in SPLIT_PATHS:
            for q in load_qrels(split)["queries"]:
                if q["category"] == "short_keyword":
                    self.assertLessEqual(len(q["query"].split()), 2, q["query_id"])

    def test_default_split_is_dev(self):
        self.assertEqual(evaluate_recall.run_benchmark.__defaults__[0], "dev")

    def test_dataset_validator_passes(self):
        with patch("sys.exit") as exit_mock:
            validate_all()
        exit_mock.assert_not_called()


class TestAnyOfRecall(unittest.TestCase):

    def setUp(self):
        self.moments = [
            {"file_id": "a.wav", "speaker": "Gary", "start_seconds": 10.0, "end_seconds": 14.0},
            {"file_id": "a.wav", "speaker": "Gary", "start_seconds": 200.0, "end_seconds": 204.0},
        ]
        self.hit_second = [{"file_id": "a.wav", "speaker": "Gary", "start_seconds": 200.0, "end_seconds": 204.0}]

    def test_any_of_counts_one_occurrence_as_full_recall(self):
        self.assertEqual(evaluate_retrieval(self.hit_second, self.moments, any_of=True)["recall@1"], 1.0)

    def test_default_requires_every_moment(self):
        self.assertEqual(evaluate_retrieval(self.hit_second, self.moments)["recall@1"], 0.5)

    def test_short_keyword_is_the_only_any_of_category(self):
        self.assertEqual(ANY_OF_CATEGORIES, {"short_keyword"})


class TestSpeakerResolutionInScorer(unittest.TestCase):

    def test_anonymous_label_is_resolved_before_comparing(self):
        gt = {"file_id": "x.wav", "speaker": "DHH", "start_seconds": 1.0, "end_seconds": 5.0}
        res = {"file_id": "x.wav", "speaker": "SPEAKER_01", "start_seconds": 1.0, "end_seconds": 5.0}
        with patch("evals.metrics.resolve_result_speaker", return_value="DHH"):
            self.assertTrue(result_matches_moment(res, gt))
        with patch("evals.metrics.resolve_result_speaker", return_value="Lex Fridman"):
            self.assertFalse(result_matches_moment(res, gt))

    def test_wrong_file_never_matches(self):
        gt = {"file_id": "x.wav", "speaker": None, "start_seconds": 1.0, "end_seconds": 5.0}
        self.assertFalse(result_matches_moment({"file_id": "y.wav", "start_seconds": 1.0, "end_seconds": 5.0}, gt))


class TestMicroRecallForShortKeywords(unittest.TestCase):

    def test_repeated_occurrences_count_as_one_moment(self):
        queries = {"queries": [{
            "query_id": "K", "category": "short_keyword", "query": "Winfield Scott", "hard_negatives": [],
            "relevant_moments": [
                {"file_id": "a.wav", "speaker": "Gary", "start_seconds": 10.0, "end_seconds": 14.0},
                {"file_id": "a.wav", "speaker": "Gary", "start_seconds": 200.0, "end_seconds": 204.0},
            ],
        }]}
        results = [{"file_id": "a.wav", "speaker": "Gary", "start_seconds": 10.0, "end_seconds": 14.0}]
        with patch.object(evaluate_recall, "load_qrels", return_value=queries), \
                patch.object(evaluate_recall, "call_api", return_value={"output": {"results": results}}), \
                patch("evals.metrics.resolve_result_speaker", side_effect=lambda r: r.get("speaker")):
            summary = evaluate_recall.run_benchmark(split="dev", modes=["hybrid"])
        self.assertEqual(summary["hybrid"]["micro_moments"]["recall@1"], 1.0)
        self.assertEqual(summary["hybrid"]["by_category"]["short_keyword"]["recall@1"], 1.0)


if __name__ == "__main__":
    unittest.main()
