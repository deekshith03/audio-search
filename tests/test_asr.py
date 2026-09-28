import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from src.pipeline import asr


def fake_segment(seg_id, start, end, text, words):
    return SimpleNamespace(
        id=seg_id, start=start, end=end, text=f" {text} ", avg_logprob=-0.12345, no_speech_prob=0.01,
        words=[SimpleNamespace(word=f" {w} ", start=s, end=e, probability=p) for w, s, e, p in words],
    )


def fake_model():
    model = MagicMock()
    model.transcribe.return_value = (
        iter([fake_segment(0, 0.0, 1.2, "Hello world.", [("Hello", 0.0, 0.5, 0.98), ("world.", 0.6, 1.2, 0.9)])]),
        SimpleNamespace(duration=2.0, language="en"),
    )
    return model


class TestSerializeSegments(unittest.TestCase):

    def test_matches_raw_asr_contract(self):
        seg = asr.serialize_segments([fake_segment(3, 1.23456, 2.5, "Hi there.", [("Hi", 1.23456, 1.5, 0.87654)])])[0]
        self.assertEqual(set(seg), {"id", "start", "end", "text", "avg_logprob", "no_speech_prob", "words"})
        self.assertEqual(seg["text"], "Hi there.")
        self.assertEqual(seg["start"], 1.235)
        self.assertEqual(seg["words"], [{"word": "Hi", "start": 1.235, "end": 1.5, "probability": 0.8765}])

    def test_segment_without_words(self):
        seg = SimpleNamespace(id=0, start=0.0, end=1.0, text="x", avg_logprob=0.0, no_speech_prob=0.0, words=None)
        self.assertEqual(asr.serialize_segments([seg])[0]["words"], [])


class TestTranscribeFile(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio = os.path.join(self.tmp.name, "clip.wav")
        sf.write(self.audio, np.zeros(32000, dtype=np.float32), 16000, subtype="PCM_16")
        self.out_dir = os.path.join(self.tmp.name, "raw_asr")

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_raw_artifact_with_config_and_cache_key(self):
        model = fake_model()
        out = asr.transcribe_file(self.audio, output_dir=self.out_dir, model=model)
        with open(out) as f:
            data = json.load(f)

        self.assertEqual(os.path.basename(out), "clip_raw.json")
        self.assertEqual(data["model"], asr.MODEL_REPO)
        self.assertEqual(data["config"]["compute_type"], "float32")
        self.assertEqual(data["config"]["device"], "cpu")
        self.assertEqual(data["text"], "Hello world.")
        self.assertEqual(len(data["segments"][0]["words"]), 2)
        self.assertTrue(data["cache_key"])
        model.transcribe.assert_called_once_with(self.audio, **asr.DECODE_OPTIONS)

    def test_valid_cache_skips_transcription(self):
        asr.transcribe_file(self.audio, output_dir=self.out_dir, model=fake_model())
        second = fake_model()
        asr.transcribe_file(self.audio, output_dir=self.out_dir, model=second)
        second.transcribe.assert_not_called()

    def test_force_recomputes(self):
        asr.transcribe_file(self.audio, output_dir=self.out_dir, model=fake_model())
        second = fake_model()
        asr.transcribe_file(self.audio, output_dir=self.out_dir, model=second, force=True)
        second.transcribe.assert_called_once()

    def test_config_change_invalidates_cache(self):
        asr.transcribe_file(self.audio, output_dir=self.out_dir, model=fake_model())
        second = fake_model()
        with patch.object(asr, "DECODE_OPTIONS", {**asr.DECODE_OPTIONS, "beam_size": 1}):
            asr.transcribe_file(self.audio, output_dir=self.out_dir, model=second)
        second.transcribe.assert_called_once()

    def test_decode_options_guard_against_known_failure_modes(self):
        self.assertEqual(asr.DECODE_OPTIONS["language"], "en")
        self.assertFalse(asr.DECODE_OPTIONS["condition_on_previous_text"])
        self.assertEqual(asr.DECODE_OPTIONS["hallucination_silence_threshold"], 2.0)
        self.assertTrue(asr.DECODE_OPTIONS["word_timestamps"])


if __name__ == "__main__":
    unittest.main()
