import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from evals import evaluate_recall
from evals.metrics import evaluate_retrieval, result_matches_moment
from evals.qrels import ANY_OF_CATEGORIES, CATEGORIES, EXPECTED_BREAKDOWN, MAX_SHORT_KEYWORD_WORDS, ONE_TIME_RESULTS, RETIRED_SPLIT_PATHS, ROOT, SPLIT_PATHS, load_qrels

BLIND_QRELS_SHA256 = "c5d8e14cad80cd2d2c9eb310cc99dd7b3cb6ec5ad33841af72a8b14d36abc984"
BLIND_QRELS_FROZEN_AS_TEST2_SHA256 = "3c47eb8436ad6536b0b8ab0491891b7d83ab1d1e1f5ef2b4a7f38b2c14a4d232"
BLIND_RESULT_SHA256 = "794f58fa4fc808ffb6a885b19c33a50f4d9f35af69f77ea6999517557b0a041a"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
from evals.validate_dataset_integrity import validate_all


class TestQrelsModule(unittest.TestCase):

    def test_split_files_exist_and_declare_their_split(self):
        for split, path in SPLIT_PATHS.items():
            self.assertTrue(os.path.exists(path), path)
            self.assertEqual(load_qrels(split)["split"], split)

    def test_breakdown_matches_expected(self):
        for split in SPLIT_PATHS:
            queries = load_qrels(split)["queries"]
            counts = {c: sum(q["category"] == c for q in queries) for c in CATEGORIES}
            self.assertEqual(counts, EXPECTED_BREAKDOWN[split])
        self.assertEqual({s: sum(EXPECTED_BREAKDOWN[s].values()) for s in SPLIT_PATHS}, {"dev": 73, "blind": 40})

    def test_only_dev_and_blind_are_active_and_blind_is_the_one_time_split(self):
        self.assertEqual(set(SPLIT_PATHS), {"dev", "blind"})
        self.assertEqual(set(ONE_TIME_RESULTS), {"blind"})
        self.assertTrue(os.path.exists(ONE_TIME_RESULTS["blind"]))

    def test_retired_splits_are_kept_but_not_active(self):
        self.assertEqual(set(RETIRED_SPLIT_PATHS), {"test", "holdout"})
        for split, path in RETIRED_SPLIT_PATHS.items():
            self.assertTrue(os.path.exists(path), path)
            self.assertNotIn(split, SPLIT_PATHS)
            self.assertNotIn(split, EXPECTED_BREAKDOWN)
            with self.assertRaises(ValueError):
                load_qrels(split)

    def test_blind_qrels_are_the_frozen_test2_file_with_only_its_name_changed(self):
        with open(SPLIT_PATHS["blind"], "rb") as f:
            data = f.read()
        self.assertEqual(sha256(data), BLIND_QRELS_SHA256)
        as_frozen = data.replace(b'"split": "blind"', b'"split": "test2"').replace(
            b"blind split (named test2 when frozen)", b"test2 split")
        self.assertEqual(sha256(as_frozen), BLIND_QRELS_FROZEN_AS_TEST2_SHA256)

    def test_blind_one_time_result_is_unchanged(self):
        with open(ONE_TIME_RESULTS["blind"], "rb") as f:
            self.assertEqual(sha256(f.read()), BLIND_RESULT_SHA256)

    def test_run_evals_refuses_blind_and_unknown_splits(self):
        script = os.path.join(ROOT, "evals", "run_evals.sh")
        for split, message in (("blind", "one-time"), ("test", "expected dev")):
            done = subprocess.run(["bash", script, "--split", split], capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(done.returncode, 64, split)
            self.assertIn(message, done.stderr, split)

    def test_rerun_of_one_time_split_is_refused_unless_forced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "final_blind.json")
            with patch.dict(evaluate_recall.ONE_TIME_RESULTS, {"blind": path}):
                evaluate_recall.refuse_rerun("blind", force=False)
                evaluate_recall.save_one_time_result("blind", {"hybrid": {"micro_r5": 0.5}}, {})
                with open(path) as f:
                    self.assertEqual(json.load(f)["report"]["hybrid"]["micro_r5"], 0.5)
                with self.assertRaises(SystemExit):
                    evaluate_recall.refuse_rerun("blind", force=False)
                evaluate_recall.refuse_rerun("blind", force=True)
                evaluate_recall.refuse_rerun("dev", force=False)

    def test_unknown_split_is_rejected(self):
        with self.assertRaises(ValueError):
            load_qrels("train")

    def test_short_keyword_queries_are_short(self):
        for split in SPLIT_PATHS:
            for q in load_qrels(split)["queries"]:
                if q["category"] == "short_keyword":
                    self.assertLessEqual(len(q["query"].split()), MAX_SHORT_KEYWORD_WORDS, q["query_id"])

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

    def test_alternative_counts_as_finding_its_moment(self):
        moment = {**self.moments[0], "alternatives": [{"file_id": "b.wav", "speaker": "Gary", "start_seconds": 50.0, "end_seconds": 54.0}]}
        on_alternative = [{"file_id": "b.wav", "speaker": "Gary", "start_seconds": 50.0, "end_seconds": 54.0}]
        self.assertEqual(evaluate_retrieval(on_alternative, [moment])["recall@1"], 1.0)
        self.assertTrue(result_matches_moment(on_alternative[0], moment))
        self.assertFalse(result_matches_moment(on_alternative[0], self.moments[0]))

    def test_alternatives_do_not_merge_distinct_moments(self):
        moment = {**self.moments[0], "alternatives": [{"file_id": "b.wav", "speaker": "Gary", "start_seconds": 50.0, "end_seconds": 54.0}]}
        on_alternative = [{"file_id": "b.wav", "speaker": "Gary", "start_seconds": 50.0, "end_seconds": 54.0}]
        self.assertEqual(evaluate_retrieval(on_alternative, [moment, self.moments[1]])["recall@5"], 0.5)

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
