import unittest

from src.pipeline.align import MIN_INTERPOLATED_WORD_SECONDS, align_config, interpolate_untimed_words, is_wildcard_word, segment_words


def timed(word, start, end, score=0.9):
    return {"word": word, "start": start, "end": end, "score": score}


def untimed(word):
    return {"word": word}


class TestInterpolateUntimedWords(unittest.TestCase):

    def test_timed_words_pass_through_in_canonical_schema(self):
        words, n = interpolate_untimed_words([timed(" hello ", 1.0, 1.4)], 0.0, 5.0)
        self.assertEqual(n, 0)
        self.assertEqual(words, [{
            "word": "hello",
            "start_seconds": 1.0,
            "end_seconds": 1.4,
            "confidence": 0.9,
            "timing_source": "wav2vec2_aligned",
        }])

    def test_single_untimed_word_fills_gap_between_timed_neighbours(self):
        words, n = interpolate_untimed_words(
            [timed("a", 1.0, 1.5), untimed("$10"), timed("b", 2.5, 3.0)], 0.0, 5.0
        )
        self.assertEqual(n, 1)
        self.assertEqual((words[1]["start_seconds"], words[1]["end_seconds"]), (1.5, 2.5))
        self.assertEqual(words[1]["timing_source"], "interpolated_fallback")
        self.assertIsNone(words[1]["confidence"])

    def test_gap_starts_at_previous_timed_word_not_segment_start(self):
        words, _ = interpolate_untimed_words(
            [timed("a", 3.0, 3.5), timed("b", 3.6, 4.0), untimed("N3"), timed("c", 5.0, 5.2)], 0.0, 9.0
        )
        self.assertEqual(words[2]["start_seconds"], 4.0)

    def test_run_of_untimed_words_is_split_evenly(self):
        words, n = interpolate_untimed_words(
            [timed("a", 1.0, 2.0), untimed("x"), untimed("y"), timed("b", 3.0, 3.5)], 0.0, 5.0
        )
        self.assertEqual(n, 2)
        self.assertEqual([(w["start_seconds"], w["end_seconds"]) for w in words[1:3]], [(2.0, 2.5), (2.5, 3.0)])

    def test_leading_untimed_word_uses_segment_start(self):
        words, _ = interpolate_untimed_words([untimed("x"), timed("a", 2.0, 2.5)], 1.0, 5.0)
        self.assertEqual((words[0]["start_seconds"], words[0]["end_seconds"]), (1.0, 2.0))

    def test_trailing_untimed_word_uses_segment_end(self):
        words, _ = interpolate_untimed_words([timed("a", 1.0, 2.0), untimed("x")], 0.0, 3.0)
        self.assertEqual((words[1]["start_seconds"], words[1]["end_seconds"]), (2.0, 3.0))

    def test_zero_gap_still_gives_positive_duration(self):
        words, _ = interpolate_untimed_words(
            [timed("a", 1.0, 2.0), untimed("x"), untimed("y"), timed("b", 2.0, 2.5)], 0.0, 5.0
        )
        for w in words[1:3]:
            self.assertAlmostEqual(w["end_seconds"] - w["start_seconds"], MIN_INTERPOLATED_WORD_SECONDS, places=3)

    def test_word_with_only_start_is_treated_as_untimed(self):
        words, n = interpolate_untimed_words([{"word": "x", "start": 1.0}], 0.0, 2.0)
        self.assertEqual(n, 1)
        self.assertEqual(words[0]["timing_source"], "interpolated_fallback")

    def test_missing_score_gives_null_confidence(self):
        words, _ = interpolate_untimed_words([{"word": "a", "start": 0.1, "end": 0.2}], 0.0, 1.0)
        self.assertIsNone(words[0]["confidence"])

    def test_empty_input(self):
        self.assertEqual(interpolate_untimed_words([], 0.0, 1.0), ([], 0))


class TestUnalignedSegments(unittest.TestCase):

    def test_aligned_words_are_used_as_they_are(self):
        words = [{"word": "Right.", "start": 1.0, "end": 1.2, "score": 0.9}]
        self.assertEqual(segment_words({"text": "Right.", "words": words}), words)

    def test_segment_with_no_words_keeps_its_text_as_untimed_words(self):
        for seg in ({"text": " Right. Okay.", "words": []}, {"text": "Right. Okay."}):
            self.assertEqual(segment_words(seg), [{"word": "Right."}, {"word": "Okay."}])

    def test_unaligned_segment_is_spread_across_the_segment_and_counted_as_fallback(self):
        words, n = interpolate_untimed_words(segment_words({"text": "Right.", "words": []}), 434.52, 434.62)
        self.assertEqual(n, 1)
        self.assertEqual(words, [{"word": "Right.", "start_seconds": 434.52, "end_seconds": 434.62,
                                  "confidence": None, "timing_source": "interpolated_fallback"}])

    def test_empty_segment_text_gives_no_words(self):
        self.assertEqual(segment_words({"text": "  ", "words": []}), [])

    def test_fallback_is_part_of_the_cache_key_config(self):
        self.assertEqual(align_config()["unaligned_segment_fallback"], "segment_text")


class TestWildcardWords(unittest.TestCase):

    def setUp(self):
        self.dictionary = {c: i for i, c in enumerate("abcdefghijklmnopqrstuvwxyz'|")}

    def test_digits_and_symbols_only_are_wildcard(self):
        self.assertTrue(is_wildcard_word("$10", self.dictionary))
        self.assertTrue(is_wildcard_word("2021.", self.dictionary))

    def test_words_with_any_vocabulary_letter_are_not_wildcard(self):
        self.assertFalse(is_wildcard_word("N3", self.dictionary))
        self.assertFalse(is_wildcard_word("GPT-4", self.dictionary))
        self.assertFalse(is_wildcard_word("hello", self.dictionary))

    def test_empty_word_is_not_wildcard(self):
        self.assertFalse(is_wildcard_word("", self.dictionary))


if __name__ == "__main__":
    unittest.main()
