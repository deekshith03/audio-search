import unittest
from unittest import mock

from evals import tune_grid

MARGIN = 1 / 19


def row(r5, r1=0.5, mrr=0.5, nm=0.5, p50=100.0, **config):
    base = {"chunker": "A-30s", "model": "bge-small", "context": True, "reranker": "bge-reranker"}
    base.update(config)
    return {"config": base, "label": tune_grid.label(base),
            "hybrid": {"micro_r5": r5, "micro_r1": r1, "macro_r5": r5, "mrr": mrr, "near_miss_reject": nm, "latency_p50_ms": p50}}


class TestCandidates(unittest.TestCase):

    def test_family_round_is_gemma_with_bge_reranker(self):
        candidates = tune_grid.family_candidates()
        self.assertEqual(len(candidates), 11)
        self.assertTrue(all(c["model"] == "gemma" and c["reranker"] == "bge-reranker" for c in candidates))
        self.assertIn({"chunker": "B-prev", "model": "gemma", "context": True, "reranker": "bge-reranker"}, candidates)

    def test_family_helpers(self):
        self.assertEqual(tune_grid.family_config("A-30s+ctx"), {"chunker": "A-30s", "model": "gemma", "context": True})
        self.assertEqual(tune_grid.family_of({"chunker": "C-512", "context": False}), "C-512")

    def test_joint_is_families_x_fusions_x_rerankers_with_reranker_outermost(self):
        candidates = tune_grid.joint_candidates()
        self.assertEqual(len(candidates), 6 * 10 * 2)
        self.assertEqual(len({tuple(sorted(c.items())) for c in candidates}), 120)
        rerankers = [c["reranker"] for c in candidates]
        self.assertEqual([r for i, r in enumerate(rerankers) if i == 0 or r != rerankers[i - 1]], list(tune_grid.RERANKER_OPTIONS))

    def test_span_options(self):
        self.assertEqual(len(tune_grid.SPAN_OPTIONS), 12)

    def test_label(self):
        self.assertEqual(tune_grid.label(row(0)["config"]), "A-30s · bge-small+ctx · bge-reranker")
        cfg = {"chunker": "D", "model": "gemma", "context": False, "reranker": None, "fusion": "convex",
               "bm25_weight": 1.0, "trigram_weight": 0.5, "dense_weight": 2.0,
               "span_extend_ratio": 0.6, "span_min_seconds": 4.0, "dedupe_gap_seconds": 2.0}
        self.assertEqual(tune_grid.label(cfg), "D · gemma · convex(1/0.5/2) · no-rerank · span 0.6/4s gap 2s")


class TestJointSelection(unittest.TestCase):

    def test_slow_configs_are_not_eligible(self):
        rows = [row(0.9, p50=5000, chunker="C-512"), row(0.6, p50=800)]
        winner, _ = tune_grid.select_joint(rows, MARGIN)
        self.assertEqual(winner["config"]["chunker"], "A-30s")

    def test_winner_reranker_must_beat_its_twin(self):
        with_rr = row(0.7, r1=0.5, mrr=0.6, p50=900)
        twin = row(0.6, r1=0.5, mrr=0.6, p50=100, reranker=None)
        winner, note = tune_grid.select_joint([with_rr, twin], MARGIN)
        self.assertIsNone(winner["config"]["reranker"])
        self.assertIn("rejected", note)

    def test_winner_reranker_kept_when_it_helps(self):
        with_rr = row(0.7, r1=0.6, mrr=0.7, p50=900)
        twin = row(0.6, r1=0.4, mrr=0.5, p50=100, reranker=None)
        winner, note = tune_grid.select_joint([with_rr, twin], MARGIN)
        self.assertEqual(winner["config"]["reranker"], "bge-reranker")

    def test_no_eligible_config_stops(self):
        with self.assertRaises(SystemExit):
            tune_grid.select_joint([row(0.9, p50=9000)], MARGIN)

    def test_best_per_family(self):
        rows = [row(0.6, chunker="B"), row(0.7, chunker="B", model="gemma"), row(0.5, chunker="D")]
        best = tune_grid.best_per_family(rows, MARGIN)
        self.assertEqual([(r["config"]["chunker"], r["hybrid"]["micro_r5"]) for r in best], [("B", 0.7), ("D", 0.5)])


class TestSpanLeaders(unittest.TestCase):

    def test_winner_plus_best_eligible_other_family(self):
        winner = row(0.8, p50=900)
        same_family = row(0.79, p50=100, reranker=None)
        other = row(0.7, p50=60, chunker="B-prev", reranker=None)
        slow_other = row(0.9, p50=5000, chunker="C-512")
        joint = {"selected": [winner["config"]], "rows": [winner, same_family, other, slow_other]}
        self.assertEqual(tune_grid.span_leaders(joint, MARGIN), [winner["config"], other["config"]])

    def test_only_winner_when_no_other_family_is_eligible(self):
        winner = row(0.8, p50=900)
        joint = {"selected": [winner["config"]], "rows": [winner, row(0.9, p50=5000, chunker="D")]}
        self.assertEqual(tune_grid.span_leaders(joint, MARGIN), [winner["config"]])


class TestSubsetMetrics(unittest.TestCase):

    def test_micro_weights_by_moments_and_macro_mrr(self):
        r = {"hybrid": {"per_query": {"A": {"r1": 1.0, "r5": 1.0, "mrr": 1.0}, "B": {"r1": 0.0, "r5": 0.5, "mrr": 0.25}}}}
        m = tune_grid.subset_metrics(r, ["A", "B", "missing"], {"A": 1, "B": 2})
        self.assertEqual(m["n"], 2)
        self.assertAlmostEqual(m["micro_r1"], 1 / 3)
        self.assertAlmostEqual(m["micro_r5"], 2 / 3)
        self.assertAlmostEqual(m["mrr"], 0.625)


class TestJointResume(unittest.TestCase):

    def test_skips_configs_already_saved(self):
        configs = tune_grid.joint_candidates(["B"])[:3]
        saved = {"rows": [{**row(0.5), "config": configs[0]}]}
        with mock.patch.object(tune_grid, "joint_candidates", return_value=configs), \
                mock.patch.object(tune_grid.os.path, "exists", return_value=True), \
                mock.patch.object(tune_grid, "load_round", return_value=saved), \
                mock.patch.object(tune_grid, "evaluate", side_effect=lambda c, d, cache: {**row(0.5, p50=100), "config": c}) as ev, \
                mock.patch.object(tune_grid, "save_round"), mock.patch.object(tune_grid, "evict_reranker"), \
                mock.patch.object(tune_grid, "total_moments", return_value=19), mock.patch("builtins.print"):
            tune_grid.run_round("joint", ["B"])
        self.assertEqual([c.args[0] for c in ev.call_args_list], configs[1:])


class TestRankRows(unittest.TestCase):

    def test_best_r5_wins_outside_margin(self):
        ordered = tune_grid.rank_rows([row(0.50, r1=0.9), row(0.70, r1=0.1)], MARGIN)
        self.assertEqual(ordered[0]["hybrid"]["micro_r5"], 0.70)

    def test_r5_within_one_moment_is_a_tie_broken_by_r1(self):
        ordered = tune_grid.rank_rows([row(0.70, r1=0.30), row(0.70 - MARGIN, r1=0.50, chunker="B")], MARGIN)
        self.assertEqual(ordered[0]["config"]["chunker"], "B")

    def test_then_mrr(self):
        ordered = tune_grid.rank_rows([row(0.7, mrr=0.60), row(0.7, mrr=0.65, chunker="D")], MARGIN)
        self.assertEqual(ordered[0]["config"]["chunker"], "D")

    def test_then_simplest(self):
        ordered = tune_grid.rank_rows([row(0.7, model="gemma"), row(0.7, model="bge-small", context=False)], MARGIN)
        self.assertEqual((ordered[0]["config"]["model"], ordered[0]["config"]["context"]), ("bge-small", False))

    def test_returns_every_row(self):
        rows = [row(x / 10) for x in range(5)]
        self.assertEqual(len(tune_grid.rank_rows(rows, MARGIN)), 5)


class TestRerankerKeepRule(unittest.TestCase):

    def rows(self, **candidate):
        return [row(0.6, r1=0.4, mrr=0.5, nm=0.5, p50=100, reranker=None), row(**{"r5": 0.6, "p50": 900, **candidate})]

    def test_keeps_reranker_that_helps_r1_within_latency(self):
        chosen = tune_grid.reranker_keep_decision(self.rows(r1=0.55, mrr=0.5), MARGIN)
        self.assertEqual(chosen["config"]["reranker"], "bge-reranker")

    def test_rejects_reranker_that_does_not_help(self):
        chosen = tune_grid.reranker_keep_decision(self.rows(r1=0.4, mrr=0.5, nm=0.5), MARGIN)
        self.assertIsNone(chosen["config"]["reranker"])

    def test_rejects_reranker_that_loses_r5(self):
        chosen = tune_grid.reranker_keep_decision(self.rows(r5=0.6 - 2 * MARGIN, r1=0.7), MARGIN)
        self.assertIsNone(chosen["config"]["reranker"])

    def test_rejects_reranker_that_is_too_slow(self):
        chosen = tune_grid.reranker_keep_decision(self.rows(r1=0.7, p50=2000), MARGIN)
        self.assertIsNone(chosen["config"]["reranker"])


class TestEvaluate(unittest.TestCase):

    def test_lexical_shared_per_chunker_and_diagnostics_recorded(self):
        metrics = row(0.5)["hybrid"]
        cache = {}
        with mock.patch.object(tune_grid, "run_mode", return_value=metrics) as run_mode, \
                mock.patch.object(tune_grid, "fused_recall_ceiling", return_value=0.9), mock.patch("builtins.print"):
            first = tune_grid.evaluate(row(0)["config"], True, cache)
            tune_grid.evaluate({**row(0)["config"], "model": "gemma"}, True, cache)
        modes = [c.args[1] for c in run_mode.call_args_list]
        self.assertEqual(modes.count("lexical"), 1)
        self.assertEqual(modes.count("dense"), 2)
        self.assertEqual(first["fused_r20"], 0.9)

    def test_no_diagnostics(self):
        with mock.patch.object(tune_grid, "run_mode", return_value=row(0.5)["hybrid"]) as run_mode, mock.patch("builtins.print"):
            result = tune_grid.evaluate(row(0)["config"], False, {})
        self.assertEqual([c.args[1] for c in run_mode.call_args_list], ["hybrid"])
        self.assertNotIn("lexical", result)


if __name__ == "__main__":
    unittest.main()
