import glob
import json
import os
import unittest

from src.search.sentences import (
    MAX_SENTENCE_SECONDS,
    MAX_SENTENCE_WORDS,
    MIN_SENTENCE_WORDS,
    ends_sentence,
    split_transcript,
    split_turn,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANONICAL_PATHS = sorted(glob.glob(os.path.join(REPO_ROOT, "dataset", "pipeline_outputs", "*_canonical.json")))


def make_turn(tokens, turn_id=1, speaker="SPEAKER_00", word_seconds=0.3, gap=0.05, pauses=None):
    """`tokens` is a list of words; `pauses` maps a word index to the silence before it."""
    pauses = pauses or {}
    words, t = [], 0.0
    for i, token in enumerate(tokens):
        t += pauses.get(i, gap if i else 0.0)
        words.append({"word": token, "start_seconds": round(t, 3), "end_seconds": round(t + word_seconds, 3)})
        t += word_seconds
    return {"turn_id": turn_id, "speaker_label": speaker, "words": words}


def texts(sentences):
    return [s.text for s in sentences]


class TestEndsSentence(unittest.TestCase):

    def test_terminal_punctuation(self):
        for word in ("done.", "why?", "wow!", 'said."', "(really.)"):
            self.assertTrue(ends_sentence(word), word)

    def test_abbreviations_and_plain_words(self):
        for word in ("Dr.", "e.g.", "U.S.", "vs.", "word", "commas,"):
            self.assertFalse(ends_sentence(word), word)


class TestSplitTurn(unittest.TestCase):

    def test_splits_on_punctuation(self):
        turn = make_turn("I like this a lot. Do you like it too? Yes I really do.".split())
        self.assertEqual(texts(split_turn(turn)), ["I like this a lot.", "Do you like it too?", "Yes I really do."])

    def test_abbreviation_does_not_split(self):
        turn = make_turn("I met Dr. Smith at the lab today.".split())
        self.assertEqual(len(split_turn(turn)), 1)

    def test_unpunctuated_run_splits_at_longest_pause(self):
        tokens = [f"w{i}" for i in range(40)]
        sentences = split_turn(make_turn(tokens, pauses={12: 0.5, 25: 0.9}))
        self.assertEqual([s.word_start for s in sentences], [0, 25])

    def test_unpunctuated_run_without_pauses_is_hard_capped(self):
        sentences = split_turn(make_turn([f"w{i}" for i in range(70)]))
        self.assertEqual([(s.word_start, s.word_end) for s in sentences], [(0, 30), (30, 60), (60, 70)])

    def test_long_by_duration_only_splits_at_pause(self):
        tokens = [f"w{i}" for i in range(20)]
        sentences = split_turn(make_turn(tokens, word_seconds=1.0, pauses={10: 0.6}))
        self.assertEqual([s.word_start for s in sentences], [0, 10])

    def test_long_by_duration_without_pause_stays_whole(self):
        sentences = split_turn(make_turn([f"w{i}" for i in range(20)], word_seconds=1.0))
        self.assertEqual(len(sentences), 1)

    def test_pause_split_avoids_creating_fragments(self):
        tokens = [f"w{i}" for i in range(35)]
        sentences = split_turn(make_turn(tokens, pauses={2: 2.0, 20: 0.5}))
        self.assertEqual([s.word_start for s in sentences], [0, 20])

    def test_short_fragment_merges_into_previous(self):
        turn = make_turn("This is a full sentence here. Yes. And another full one here.".split())
        self.assertEqual(texts(split_turn(turn)), ["This is a full sentence here. Yes.", "And another full one here."])

    def test_short_leading_fragment_merges_into_next(self):
        turn = make_turn("Right. So this is the actual point.".split())
        self.assertEqual(texts(split_turn(turn)), ["Right. So this is the actual point."])

    def test_single_word_turn_is_one_sentence(self):
        self.assertEqual(texts(split_turn(make_turn(["Yeah."]))), ["Yeah."])

    def test_empty_turn(self):
        self.assertEqual(split_turn(make_turn([])), [])

    def test_timestamps_come_from_first_and_last_word(self):
        turn = make_turn("One two three four. Five six seven eight.".split())
        first, second = split_turn(turn)
        self.assertEqual((first.start_s, first.end_s), (turn["words"][0]["start_seconds"], turn["words"][3]["end_seconds"]))
        self.assertEqual(second.start_s, turn["words"][4]["start_seconds"])

    def test_carries_turn_and_speaker(self):
        (sentence,) = split_turn(make_turn("Hello there my friend.".split(), turn_id=7, speaker="SPEAKER_01"))
        self.assertEqual((sentence.turn_id, sentence.speaker_label), (7, "SPEAKER_01"))


class TestGoldenTranscripts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.transcripts = []
        for path in CANONICAL_PATHS:
            with open(path, encoding="utf-8") as f:
                canonical = json.load(f)
            cls.transcripts.append((canonical, split_transcript(canonical)))

    def test_golden_set_present(self):
        self.assertEqual(len(self.transcripts), 7)

    def test_sentences_partition_every_turn_exactly(self):
        for canonical, sentences in self.transcripts:
            for turn in canonical["turns"]:
                spans = [(s.word_start, s.word_end) for s in sentences if s.turn_id == turn["turn_id"]]
                covered = [i for a, b in spans for i in range(a, b)]
                self.assertEqual(covered, list(range(len(turn["words"]))), (canonical["file_id"], turn["turn_id"]))

    def test_sentence_lengths_are_bounded(self):
        for canonical, sentences in self.transcripts:
            for s in sentences:
                words = s.word_end - s.word_start
                self.assertLessEqual(words, MAX_SENTENCE_WORDS + MIN_SENTENCE_WORDS, (canonical["file_id"], s.text))
                self.assertLessEqual(s.end_s - s.start_s, MAX_SENTENCE_SECONDS * 1.5, (canonical["file_id"], s.text))

    def test_unpunctuated_file_still_gets_short_sentences(self):
        canonical, sentences = next(t for t in self.transcripts if t[0]["file_id"].startswith("audio_02"))
        self.assertGreater(len(sentences), 80)


if __name__ == "__main__":
    unittest.main()
