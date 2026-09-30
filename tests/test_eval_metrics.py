"""
Comprehensive Unit Tests for Evaluation Metrics, Temporal Matching, and Gates.

Strictly tests:
- Temporal IoU exact and partial calculation (e.g. 1/3)
- Strict positive overlap requirement (rejecting adjacent non-overlapping intervals)
- Pathological prediction rejection (max duration, minimal coverage)
- Speaker attribution matching across micro and macro recall
- Promptfoo JSON and YAML complete structural parity
- Near-Miss Hard Negative Rejection logic (Rank 1 vs below Rank 1)
- Gate check fail-closed enforcement (missing modes, NaN/Inf, strict > ablation proof)
"""

import unittest
import sys
import os
import json
import yaml
import math

# Ensure evals directory is in sys.path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evals"))

from evals.metrics import compute_temporal_iou, is_temporal_match, evaluate_retrieval
from evals.assertions import check_temporal_match, get_assert
from evals.evaluate_recall import check_gate
from evals.qrels import SPLIT_PATHS


class TestEvaluationMetrics(unittest.TestCase):

    def test_temporal_iou_exact_and_partial(self):
        # Exact match -> IoU = 1.0
        self.assertAlmostEqual(compute_temporal_iou(10.0, 20.0, 10.0, 20.0), 1.0)
        # Disjoint -> IoU = 0.0
        self.assertAlmostEqual(compute_temporal_iou(0.0, 10.0, 20.0, 30.0), 0.0)
        # Adjacent non-overlapping -> IoU = 0.0
        self.assertAlmostEqual(compute_temporal_iou(0.0, 5.0, 5.0, 10.0), 0.0)
        # Partial overlap: [0, 10] and [5, 15] -> overlap=5, union=15 -> IoU = 1/3
        self.assertAlmostEqual(compute_temporal_iou(0.0, 10.0, 5.0, 15.0), 1.0 / 3.0)

    def test_strict_positive_overlap_required(self):
        # Even if start_delta is small (1.0s), if overlap is zero or negative, it MUST FAIL!
        match_pred, delta, iou = is_temporal_match(
            pred_start=0.0, pred_end=1.0, gt_start=1.0, gt_end=2.0, tolerance_seconds=5.0
        )
        self.assertFalse(match_pred, "Adjacent non-overlapping intervals must not match!")

        # Overlapping interval with delta <= 5.0s should pass
        match_pred2, _, _ = is_temporal_match(
            pred_start=1.0, pred_end=10.0, gt_start=2.0, gt_end=9.0, tolerance_seconds=5.0
        )
        self.assertTrue(match_pred2, "Overlapping interval with start delta <= 5s must match")

    def test_pathological_prediction_rejection(self):
        # Prediction spanning 10,000 seconds must be rejected as unlocalized
        valid = check_temporal_match(
            pred_start=0.0, pred_end=10000.0, gt_start=0.0, gt_end=10.0, max_duration=120.0
        )
        self.assertFalse(valid, "Excessively long prediction chunk must be rejected")

        # Inverted interval (end < start) must be rejected
        valid_inverted = check_temporal_match(
            pred_start=50.0, pred_end=10.0, gt_start=10.0, gt_end=20.0
        )
        self.assertFalse(valid_inverted, "Inverted intervals must be rejected")

    def test_speaker_attribution_matching(self):
        gt = [{"file_id": "audio_01.wav", "speaker": "DHH", "start_seconds": 100.0, "end_seconds": 120.0}]
        
        # Wrong speaker must NOT count as a hit
        res_wrong_spk = [{"file_id": "audio_01.wav", "speaker": "Lex Fridman", "start_seconds": 102.0, "end_seconds": 118.0}]
        eval_wrong = evaluate_retrieval(res_wrong_spk, gt, k_values=[1, 5])
        self.assertEqual(eval_wrong["recall@1"], 0.0)
        self.assertEqual(eval_wrong["recall@5"], 0.0)

        # Correct speaker must count as a hit
        res_correct = [{"file_id": "audio_01.wav", "speaker": "DHH", "start_seconds": 102.0, "end_seconds": 118.0}]
        eval_correct = evaluate_retrieval(res_correct, gt, k_values=[1, 5])
        self.assertEqual(eval_correct["recall@1"], 1.0)
        self.assertEqual(eval_correct["recall@5"], 1.0)

    def test_json_and_yaml_parity(self):
        for split in SPLIT_PATHS:
            with self.subTest(split=split):
                with open(f"promptfooconfig.{split}.json", "r", encoding="utf-8") as fj:
                    d_json = json.load(fj)
                with open(f"promptfooconfig.{split}.yaml", "r", encoding="utf-8") as fy:
                    d_yaml = yaml.safe_load(fy)

                self.assertEqual(d_json, d_yaml, f"promptfooconfig.{split}.json and .yaml must be structurally identical")

                # hard_negatives must be an explicit list: empty except for near-miss queries
                for i, t in enumerate(d_yaml["tests"]):
                    cat = t["vars"]["category"]
                    hn = t["vars"]["hard_negatives"]
                    self.assertIsInstance(hn, list, f"{split} test {i} ({t['description']}) hard_negatives must be a list")
                    if cat == "near_miss":
                        self.assertGreater(len(hn), 0, f"{split} near-miss test {i} must have at least one hard negative")
                    else:
                        self.assertEqual(len(hn), 0, f"{split} test {i} ({t['description']}) hard_negatives must be empty")

    def test_configs_match_qrels(self):
        from evals.qrels import load_qrels
        for split in SPLIT_PATHS:
            with self.subTest(split=split):
                with open(f"promptfooconfig.{split}.yaml", "r", encoding="utf-8") as fy:
                    cfg = yaml.safe_load(fy)
                self.assertEqual([t["vars"]["query_id"] for t in cfg["tests"]], [q["query_id"] for q in load_qrels(split)["queries"]])

    def test_near_miss_hard_negative_rejection(self):
        context = {
            "vars": {
                "query_id": "NM-01",
                "category": "near_miss",
                "expected_moments": [
                    {"file_id": "audio_02.wav", "speaker": "Jonathan", "start_seconds": 442.2, "end_seconds": 533.8}
                ],
                "hard_negatives": [
                    {"file_id": "audio_01.wav", "speaker": "DHH", "start_seconds": 491.0, "end_seconds": 531.0, "reason": "AI coding distractor"}
                ]
            }
        }

        # Case A: Hard negative ranked at Rank #1 -> MUST FAIL
        out_fail = {
            "results": [
                {"file_id": "audio_01.wav", "speaker": "DHH", "start_seconds": 492.0, "end_seconds": 530.0},
                {"file_id": "audio_02.wav", "speaker": "Jonathan", "start_seconds": 442.2, "end_seconds": 533.8}
            ]
        }
        assert_res = get_assert(out_fail, context)
        self.assertFalse(assert_res["pass"], "Ranking hard negative at Rank 1 must fail near-miss check")
        self.assertIn("NEAR-MISS FAILURE", assert_res["reason"])

        # Case B: True target at Rank #1, hard negative at Rank #2 -> MUST PASS
        out_pass = {
            "results": [
                {"file_id": "audio_02.wav", "speaker": "Jonathan", "start_seconds": 442.2, "end_seconds": 533.8},
                {"file_id": "audio_01.wav", "speaker": "DHH", "start_seconds": 492.0, "end_seconds": 530.0}
            ]
        }
        assert_pass = get_assert(out_pass, context)
        self.assertTrue(assert_pass["pass"], "Having true target at Rank 1 must pass near-miss check")

    def test_gate_check_enforcement(self):
        # 1. Missing mode must fail closed
        with self.assertRaises(SystemExit):
            check_gate({"hybrid": {"micro_moments": {"recall@1": 0.65, "recall@5": 0.90}}})

        # 2. Missing metric keys must fail closed
        with self.assertRaises(SystemExit):
            check_gate({
                "hybrid": {"micro_moments": {"recall@5": 0.90}},
                "lexical": {"micro_moments": {"recall@5": 0.70}},
                "dense": {"micro_moments": {"recall@5": 0.70}}
            })

        # 3. NaN or Inf metrics must fail closed
        with self.assertRaises(SystemExit):
            check_gate({
                "hybrid": {"micro_moments": {"recall@1": float("nan"), "recall@5": 0.90}},
                "lexical": {"micro_moments": {"recall@5": 0.70}},
                "dense": {"micro_moments": {"recall@5": 0.70}}
            })
        with self.assertRaises(SystemExit):
            check_gate({
                "hybrid": {"micro_moments": {"recall@1": 0.65, "recall@5": float("inf")}},
                "lexical": {"micro_moments": {"recall@5": 0.70}},
                "dense": {"micro_moments": {"recall@5": 0.70}}
            })

        # 4. Failing recall thresholds
        failing_summary = {
            "hybrid": {"micro_moments": {"recall@1": 0.50, "recall@5": 0.80}},
            "lexical": {"micro_moments": {"recall@5": 0.80}},
            "dense": {"micro_moments": {"recall@5": 0.70}}
        }
        with self.assertRaises(SystemExit):
            check_gate(failing_summary)

        # 5. Passing summary: Recall@5 >= 0.85, Recall@1 >= 0.60, Hybrid strictly > Lexical and Dense
        passing_summary = {
            "hybrid": {"micro_moments": {"recall@1": 0.65, "recall@5": 0.90}},
            "lexical": {"micro_moments": {"recall@5": 0.82}},
            "dense": {"micro_moments": {"recall@5": 0.78}}
        }
        check_gate(passing_summary)


if __name__ == "__main__":
    unittest.main()
