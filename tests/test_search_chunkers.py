import glob
import json
import os
import unittest
from collections import defaultdict

from src.search.chunkers import (
    C_MAX_TOKENS,
    C_OVERLAP_TOKENS,
    CHUNK_CONFIGS,
    CONTEXT_CONFIGS,
    CONTEXT_MAX_WORDS,
    MAX_WINDOW_STRETCH,
    WINDOW_SECONDS,
    build_chunks,
    embedding_input,
    sentence_windows,
    token_windows,
)
from src.search.sentences import MAX_SENTENCE_WORDS, MIN_SENTENCE_WORDS, Sentence, split_transcript

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_PATHS = sorted(glob.glob(os.path.join(REPO_ROOT, "dataset", "pipeline_outputs", "*_canonical.json")))


def one_token_per_word(word):
    return 1


def sentence_at(start, end, word_start=0, word_end=1):
    return Sentence(1, "SPEAKER_00", word_start, word_end, start, end, "x")


def timed_words(n, seconds_each=1.0):
    return [{"word": f"w{i}", "start_seconds": i * seconds_each, "end_seconds": (i + 1) * seconds_each} for i in range(n)]


def conversation():
    """Three turns: A asks, B answers at length, A reacts."""
    def turn(turn_id, speaker, tokens, offset):
        words = [{"word": t, "start_seconds": offset + i, "end_seconds": offset + i + 0.8} for i, t in enumerate(tokens)]
        return {"turn_id": turn_id, "speaker_label": speaker, "words": words}

    question = "Did the project ship on time last year?".split()
    answer = ("Unfortunately no it did not. " * 12).split()
    reaction = "Oh that is a real shame then.".split()
    return {
        "file_id": "t.wav",
        "turns": [
            turn(1, "SPEAKER_00", question, 0),
            turn(2, "SPEAKER_01", answer, 10),
            turn(3, "SPEAKER_00", reaction, 80),
        ],
    }


class TestSentenceWindows(unittest.TestCase):

    def test_packs_sentences_with_one_sentence_overlap(self):
        sentences = [sentence_at(i * 10, i * 10 + 9) for i in range(7)]
        self.assertEqual(sentence_windows(sentences, 30.0), [[0, 1, 2], [2, 3, 4], [4, 5, 6]])

    def test_sentence_longer_than_window_stands_alone(self):
        sentences = [sentence_at(0, 40), sentence_at(41, 45), sentence_at(46, 50)]
        self.assertEqual(sentence_windows(sentences, 30.0), [[0], [1, 2]])

    def test_short_tail_is_folded_into_previous_window(self):
        sentences = [sentence_at(i * 10, i * 10 + 9) for i in range(3)] + [sentence_at(30, 34)]
        self.assertEqual(sentence_windows(sentences, 30.0), [[0, 1, 2, 3]])

    def test_every_later_window_adds_a_new_sentence(self):
        sentences = [sentence_at(i * 20, i * 20 + 19) for i in range(5)]
        windows = sentence_windows(sentences, 30.0)
        for prev, cur in zip(windows, windows[1:]):
            self.assertGreater(cur[-1], prev[-1])

    def test_single_sentence(self):
        self.assertEqual(sentence_windows([sentence_at(0, 5)], 30.0), [[0]])


class TestTokenWindows(unittest.TestCase):

    def test_fits_in_one_window(self):
        self.assertEqual(token_windows(timed_words(100), one_token_per_word), [range(0, 100)])

    def test_windows_respect_limit_and_overlap(self):
        windows = token_windows(timed_words(1200), one_token_per_word)
        self.assertEqual(windows[0], range(0, C_MAX_TOKENS))
        self.assertEqual(windows[1].start, C_MAX_TOKENS - C_OVERLAP_TOKENS)
        self.assertEqual(windows[-1].stop, 1200)
        self.assertTrue(all(len(w) <= C_MAX_TOKENS for w in windows))

    def test_word_longer_than_limit_still_progresses(self):
        windows = token_windows(timed_words(3), lambda w: C_MAX_TOKENS + 5)
        self.assertEqual(windows, [range(0, 1), range(1, 2), range(2, 3)])


class TestBuildChunksOnConversation(unittest.TestCase):

    def setUp(self):
        self.canonical = conversation()
        self.sentences = split_transcript(self.canonical)
        self.chunks = build_chunks(self.canonical, self.sentences, count_tokens=one_token_per_word)

    def by_config(self, config):
        return [c for c in self.chunks if c.chunker == config]

    def test_every_config_is_built(self):
        self.assertEqual({c.chunker for c in self.chunks}, set(CHUNK_CONFIGS))

    def test_d_is_one_chunk_per_turn(self):
        self.assertEqual([c.turn_id for c in self.by_config("D")], [1, 2, 3])

    def test_b_is_one_chunk_per_sentence(self):
        self.assertEqual([c.text for c in self.by_config("B")], [s.text for s in self.sentences])
        self.assertEqual([c.sentence_indexes for c in self.by_config("B")], [(i,) for i in range(len(self.sentences))])

    def test_context_is_previous_turn_of_other_speaker(self):
        answer_chunks = [c for c in self.by_config("B") if c.turn_id == 2]
        self.assertTrue(all(c.context_text == "Did the project ship on time last year?" for c in answer_chunks))

    def test_first_turn_has_no_context(self):
        self.assertIsNone(self.by_config("A-30s")[0].context_text)

    def test_b_prev_context_is_question_then_previous_sentence(self):
        answer = [c for c in self.by_config("B-prev") if c.turn_id == 2]
        b = [c for c in self.by_config("B") if c.turn_id == 2]
        self.assertEqual([c.text for c in answer], [c.text for c in b])
        self.assertEqual(answer[0].context_text, "Did the project ship on time last year?")
        self.assertEqual(answer[1].context_text, f"Did the project ship on time last year?\n\n{answer[0].text}")

    def test_b_prev_first_turn_uses_previous_sentence_only(self):
        canonical = {"file_id": "x.wav", "turns": [{"turn_id": 1, "speaker_label": "SPEAKER_00", "words": [
            {"word": w, "start_seconds": i, "end_seconds": i + 0.5}
            for i, w in enumerate("First point is here now. Second point is here too.".split())]}]}
        chunks = build_chunks(canonical, split_transcript(canonical), configs=["B-prev"])
        self.assertEqual([c.context_text for c in chunks], [None, "First point is here now."])

    def test_context_only_for_a_and_b(self):
        for c in self.chunks:
            if c.chunker not in CONTEXT_CONFIGS:
                self.assertIsNone(c.context_text)

    def test_embedding_input_prepends_context_only_when_asked(self):
        chunk = next(c for c in self.by_config("B") if c.turn_id == 2)
        self.assertEqual(embedding_input(chunk.text, chunk.context_text, True), f"{chunk.context_text}\n\n{chunk.text}")
        self.assertEqual(embedding_input(chunk.text, chunk.context_text, False), chunk.text)
        self.assertEqual(embedding_input("t", None, True), "t")

    def test_unknown_config_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown chunk config"):
            build_chunks(self.canonical, self.sentences, configs=["E"])

    def test_c_requires_token_counter(self):
        with self.assertRaisesRegex(ValueError, "token counter"):
            build_chunks(self.canonical, self.sentences, configs=["C-512"])


class TestGoldenChunkInvariants(unittest.TestCase):
    """Properties every config must hold on all seven golden transcripts."""

    @classmethod
    def setUpClass(cls):
        cls.files = []
        for path in CANONICAL_PATHS:
            with open(path, encoding="utf-8") as f:
                canonical = json.load(f)
            sentences = split_transcript(canonical)
            cls.files.append((canonical, sentences, build_chunks(canonical, sentences, count_tokens=one_token_per_word)))

    def test_chunks_stay_inside_their_turn_and_speaker(self):
        for canonical, _, chunks in self.files:
            turns = {t["turn_id"]: t for t in canonical["turns"]}
            for c in chunks:
                turn = turns[c.turn_id]
                self.assertEqual(c.speaker_label, turn["speaker_label"])
                self.assertGreaterEqual(c.start_s, turn["words"][0]["start_seconds"])
                self.assertLessEqual(c.end_s, turn["words"][-1]["end_seconds"])
                self.assertGreaterEqual(c.end_s, c.start_s)

    def test_every_config_covers_every_word(self):
        for canonical, _, chunks in self.files:
            expected = " ".join(w["word"] for t in canonical["turns"] for w in t["words"]).split()
            for config in CHUNK_CONFIGS:
                config_chunks = [c for c in chunks if c.chunker == config]
                joined = set(w for c in config_chunks for w in c.text.split())
                self.assertEqual(joined, set(expected), (canonical["file_id"], config))
                self.assertEqual([c.start_s for c in config_chunks], sorted(c.start_s for c in config_chunks))

    def test_sentence_indexes_point_at_sentences_inside_the_chunk(self):
        for canonical, sentences, chunks in self.files:
            for c in chunks:
                self.assertTrue(c.sentence_indexes, (canonical["file_id"], c.chunker, c.text))
                for i in c.sentence_indexes:
                    s = sentences[i]
                    self.assertEqual(s.turn_id, c.turn_id)
                    self.assertLess(s.start_s, c.end_s + 1e-6)
                    self.assertGreater(s.end_s, c.start_s - 1e-6)

    def test_a_windows_near_their_target_length(self):
        # One- and two-sentence windows may run long: a single long sentence, or the overlap
        # sentence plus the one new sentence every window must add.
        for canonical, sentences, chunks in self.files:
            for c in chunks:
                if c.chunker in WINDOW_SECONDS and len(c.sentence_indexes) > 2:
                    limit = WINDOW_SECONDS[c.chunker] * MAX_WINDOW_STRETCH + 1e-6
                    self.assertLessEqual(c.end_s - c.start_s, limit, (canonical["file_id"], c.chunker))

    def test_context_is_capped(self):
        for _, _, chunks in self.files:
            for c in chunks:
                if c.context_text:
                    limit = CONTEXT_MAX_WORDS + (MAX_SENTENCE_WORDS + MIN_SENTENCE_WORDS if c.chunker == "B-prev" else 0)
                    self.assertLessEqual(len(c.context_text.split()), limit)

    def test_c_windows_respect_token_limit(self):
        for _, _, chunks in self.files:
            for c in chunks:
                if c.chunker == "C-512":
                    self.assertLessEqual(len(c.text.split()), C_MAX_TOKENS)

    def test_config_chunk_counts_are_ordered_by_granularity(self):
        counts = defaultdict(int)
        for _, _, chunks in self.files:
            for c in chunks:
                counts[c.chunker] += 1
        self.assertGreater(counts["B"], counts["A-15s"])
        self.assertGreater(counts["A-15s"], counts["A-30s"])
        self.assertGreaterEqual(counts["A-30s"], counts["A-45s"])
        self.assertGreaterEqual(counts["A-45s"], counts["D"])
        self.assertEqual(counts["D"], sum(len(c["turns"]) for c, _, _ in self.files))


if __name__ == "__main__":
    unittest.main()
