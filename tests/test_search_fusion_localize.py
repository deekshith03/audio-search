import unittest

from src.search.fusion import convex, fuse, min_max, rrf
from src.search.localize import ScoredSentence, blend, dedupe, normalize, overlaps, select_span


class TestRrf(unittest.TestCase):

    def test_items_high_in_both_lists_win(self):
        lists = {"a": [(1, 9.0), (2, 8.0), (3, 7.0)], "b": [(2, 0.9), (1, 0.8), (4, 0.7)]}
        self.assertEqual([i for i, _ in rrf(lists, {"a": 1.0, "b": 1.0})][:2], [1, 2])

    def test_score_formula(self):
        (item, score), = rrf({"a": [(5, 1.0)]}, {"a": 2.0}, k=60)
        self.assertEqual(item, 5)
        self.assertAlmostEqual(score, 2.0 / 61)

    def test_weights_shift_the_ranking(self):
        lists = {"a": [(1, 1.0)], "b": [(2, 1.0)]}
        self.assertEqual(rrf(lists, {"a": 1.0, "b": 3.0})[0][0], 2)

    def test_zero_or_missing_weight_ignores_list(self):
        lists = {"a": [(1, 1.0)], "b": [(2, 1.0)]}
        self.assertEqual([i for i, s in rrf(lists, {"a": 1.0}) if s > 0], [1])

    def test_ties_break_by_id(self):
        self.assertEqual([i for i, _ in rrf({"a": [(7, 1.0)], "b": [(3, 1.0)]}, {"a": 1.0, "b": 1.0})], [3, 7])

    def test_empty(self):
        self.assertEqual(rrf({}, {}), [])


class TestConvex(unittest.TestCase):

    def test_min_max(self):
        self.assertEqual(min_max([(1, 10.0), (2, 5.0), (3, 0.0)]), {1: 1.0, 2: 0.5, 3: 0.0})
        self.assertEqual(min_max([(1, 3.0), (2, 3.0)]), {1: 1.0, 2: 1.0})
        self.assertEqual(min_max([]), {})

    def test_uses_score_magnitudes_not_just_ranks(self):
        lists = {"a": [(1, 10.0), (2, 9.9), (3, 0.0)], "b": [(3, 1.0), (2, 0.9), (1, 0.0)]}
        self.assertEqual(convex(lists, {"a": 0.5, "b": 0.5})[0][0], 2)

    def test_weights_are_normalized(self):
        lists = {"a": [(1, 1.0), (2, 0.0)]}
        self.assertEqual(convex(lists, {"a": 4.0}), [(1, 1.0), (2, 0.0)])

    def test_missing_item_contributes_zero(self):
        scores = dict(convex({"a": [(1, 1.0), (2, 0.0)], "b": [(2, 1.0), (3, 0.0)]}, {"a": 1.0, "b": 1.0}))
        self.assertEqual(scores, {1: 0.5, 2: 0.5, 3: 0.0})

    def test_fuse_dispatch(self):
        lists = {"a": [(1, 1.0)]}
        self.assertEqual(fuse("rrf", lists, {"a": 1.0}), rrf(lists, {"a": 1.0}))
        self.assertEqual(fuse("convex", lists, {"a": 1.0}), convex(lists, {"a": 1.0}))
        with self.assertRaisesRegex(ValueError, "unknown fusion"):
            fuse("max", lists, {})


def sentences(*specs):
    """specs: (start, end, score) tuples; ids are 1..n."""
    return [ScoredSentence(i, s, e, f"s{i}", sc) for i, (s, e, sc) in enumerate(specs, start=1)]


class TestNormalizeAndBlend(unittest.TestCase):

    def test_normalize(self):
        self.assertEqual(normalize([2.0, 4.0, None]), [0.0, 1.0, 0.0])
        self.assertEqual(normalize([None, None]), [0.0, 0.0])
        self.assertEqual(normalize([0.0, 0.0]), [0.0, 0.0])
        self.assertEqual(normalize([0.5, 0.5]), [1.0, 1.0])

    def test_blend_weights_signals(self):
        signals = {"dense": [0.1, 0.9], "keyword": [1.0, 0.0]}
        self.assertEqual(blend(signals, {"dense": 3.0, "keyword": 1.0}), [0.25, 0.75])

    def test_blend_single_signal(self):
        self.assertEqual(blend({"keyword": [0.0, 2.0]}, {"keyword": 1.0}), [0.0, 1.0])


class TestSelectSpan(unittest.TestCase):

    def test_grows_toward_strong_neighbours(self):
        window = sentences((0, 4, 0.1), (4, 8, 0.9), (8, 12, 1.0), (12, 16, 0.2))
        self.assertEqual(select_span(window, {1, 2, 3, 4}), (1, 2))

    def test_stops_at_max_sentences(self):
        window = sentences(*[(i * 3, i * 3 + 3, 1.0) for i in range(6)])
        lo, hi = select_span(window, {1, 2, 3, 4, 5, 6})
        self.assertEqual(hi - lo + 1, 3)

    def test_stops_at_max_seconds(self):
        window = sentences((0, 12, 1.0), (12, 24, 1.0))
        self.assertEqual(select_span(window, {1, 2}, max_seconds=20.0), (0, 0))

    def test_short_span_extends_even_to_weak_neighbour(self):
        window = sentences((0, 2, 1.0), (2, 6, 0.1))
        self.assertEqual(select_span(window, {1, 2}, min_seconds=4.0), (0, 1))

    def test_long_enough_span_does_not_take_weak_neighbour(self):
        window = sentences((0, 6, 1.0), (6, 10, 0.1))
        self.assertEqual(select_span(window, {1, 2}), (0, 0))

    def test_seed_is_from_chunk_but_growth_may_use_neighbours(self):
        window = sentences((0, 5, 1.0), (5, 10, 0.95), (10, 15, 0.3))
        self.assertEqual(select_span(window, {2, 3}), (0, 1))

    def test_no_eligible_sentence_raises(self):
        with self.assertRaises(ValueError):
            select_span(sentences((0, 5, 1.0)), {99})


class TestDedupe(unittest.TestCase):

    def test_overlaps(self):
        self.assertTrue(overlaps(0, 10, 9, 20))
        self.assertFalse(overlaps(0, 10, 10, 20))

    def test_drops_later_overlapping_span_in_same_file(self):
        spans = [("f1", 0, 10), ("f1", 5, 15), ("f2", 5, 15), ("f1", 10, 20)]
        self.assertEqual(dedupe(spans), [0, 2, 3])

    def test_empty(self):
        self.assertEqual(dedupe([]), [])


if __name__ == "__main__":
    unittest.main()
