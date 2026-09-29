import glob
import json
import os
import unittest

from src.search.chunkers import MAX_WINDOW_STRETCH, WINDOW_SECONDS, build_chunks, sentence_windows
from src.search.sentences import Sentence, split_transcript

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_PATHS = sorted(glob.glob(os.path.join(REPO_ROOT, "dataset", "pipeline_outputs", "*_canonical.json")))


def sentence_at(start, end, word_start=0, word_end=1):
    return Sentence(1, "SPEAKER_00", word_start, word_end, start, end, "x")


def conversation():
    """Three turns: A asks, B answers at length, A reacts."""
    def turn(turn_id, speaker, tokens, offset):
        words = [{"word": t, "start_seconds": offset + i, "end_seconds": offset + i + 0.8} for i, t in enumerate(tokens)]
        return {"turn_id": turn_id, "speaker_label": speaker, "words": words}

    return {
        "file_id": "t.wav",
        "turns": [
            turn(1, "SPEAKER_00", "Did the project ship on time last year?".split(), 0),
            turn(2, "SPEAKER_01", ("Unfortunately no it did not. " * 12).split(), 10),
            turn(3, "SPEAKER_00", "Oh that is a real shame then.".split(), 80),
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
        self.assertEqual(sentence_windows([sentence_at(0, 5)]), [[0]])


class TestBuildChunksOnConversation(unittest.TestCase):

    def setUp(self):
        self.canonical = conversation()
        self.sentences = split_transcript(self.canonical)
        self.chunks = build_chunks(self.canonical, self.sentences)

    def test_chunks_never_cross_turns(self):
        self.assertEqual(sorted({c.turn_id for c in self.chunks}), [1, 2, 3])
        for c in self.chunks:
            for i in c.sentence_indexes:
                self.assertEqual(self.sentences[i].turn_id, c.turn_id)

    def test_long_turn_splits_into_overlapping_windows(self):
        answer = [c for c in self.chunks if c.turn_id == 2]
        self.assertGreater(len(answer), 1)
        for prev, cur in zip(answer, answer[1:]):
            self.assertEqual(prev.sentence_indexes[-1], cur.sentence_indexes[0])

    def test_empty_turn_is_skipped(self):
        canonical = {"file_id": "e.wav", "turns": [{"turn_id": 1, "speaker_label": "SPEAKER_00", "words": []}]}
        self.assertEqual(build_chunks(canonical, split_transcript(canonical)), [])


class TestGoldenChunkInvariants(unittest.TestCase):
    """Properties the chunker must hold on all seven golden transcripts."""

    @classmethod
    def setUpClass(cls):
        cls.files = []
        for path in CANONICAL_PATHS:
            with open(path, encoding="utf-8") as f:
                canonical = json.load(f)
            sentences = split_transcript(canonical)
            cls.files.append((canonical, sentences, build_chunks(canonical, sentences)))

    def test_chunks_stay_inside_their_turn_and_speaker(self):
        for canonical, _, chunks in self.files:
            turns = {t["turn_id"]: t for t in canonical["turns"]}
            for c in chunks:
                turn = turns[c.turn_id]
                self.assertEqual(c.speaker_label, turn["speaker_label"])
                self.assertGreaterEqual(c.start_s, turn["words"][0]["start_seconds"])
                self.assertLessEqual(c.end_s, turn["words"][-1]["end_seconds"])
                self.assertGreaterEqual(c.end_s, c.start_s)

    def test_every_word_is_covered_and_order_is_monotonic(self):
        for canonical, _, chunks in self.files:
            expected = set(" ".join(w["word"] for t in canonical["turns"] for w in t["words"]).split())
            self.assertEqual({w for c in chunks for w in c.text.split()}, expected, canonical["file_id"])
            self.assertEqual([c.start_s for c in chunks], sorted(c.start_s for c in chunks))

    def test_sentence_indexes_point_at_sentences_inside_the_chunk(self):
        for canonical, sentences, chunks in self.files:
            for c in chunks:
                self.assertTrue(c.sentence_indexes, (canonical["file_id"], c.text))
                for i in c.sentence_indexes:
                    s = sentences[i]
                    self.assertEqual(s.turn_id, c.turn_id)
                    self.assertLess(s.start_s, c.end_s + 1e-6)
                    self.assertGreater(s.end_s, c.start_s - 1e-6)

    def test_windows_near_their_target_length(self):
        # One- and two-sentence windows may run long: a single long sentence, or the overlap
        # sentence plus the one new sentence every window must add.
        for canonical, _, chunks in self.files:
            for c in chunks:
                if len(c.sentence_indexes) > 2:
                    self.assertLessEqual(c.end_s - c.start_s, WINDOW_SECONDS * MAX_WINDOW_STRETCH + 1e-6, canonical["file_id"])


if __name__ == "__main__":
    unittest.main()
