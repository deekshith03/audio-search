import io
import os
import tempfile
import unittest

import numpy as np
import soundfile as sf

from src.pipeline.speaker_samples import extract_clip, select_speaker_samples, speaking_time


def turn(turn_id, speaker, start, end, confidence=0.9, short=False, word_step=0.5):
    words = []
    t = start
    while t < end - 1e-9:
        words.append({"word": f"w{turn_id}", "start_seconds": round(t, 3), "end_seconds": round(min(t + word_step * 0.8, end), 3), "confidence": confidence})
        t += word_step
    return {"turn_id": turn_id, "speaker_label": speaker, "start_seconds": start, "end_seconds": end,
            "is_short_turn": short, "text": " ".join(w["word"] for w in words), "words": words}


def canonical(turns, duration=300.0):
    return {"audio_duration_seconds": duration, "speaker_labels": ["SPEAKER_00", "SPEAKER_01"], "turns": turns}


class TestSelectSpeakerSamples(unittest.TestCase):

    def test_three_per_speaker_spread_across_recording(self):
        turns = []
        for i, start in enumerate(range(0, 300, 20)):
            turns.append(turn(i, "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01", start, start + 10))
        samples = select_speaker_samples(canonical(turns))

        for speaker in ("SPEAKER_00", "SPEAKER_01"):
            picks = samples[speaker]
            self.assertEqual(len(picks), 3)
            self.assertTrue(all(p["speaker_label"] == speaker for p in picks))
            thirds = {int(p["start_seconds"] // 100) for p in picks}
            self.assertEqual(thirds, {0, 1, 2})
            self.assertEqual(picks, sorted(picks, key=lambda p: p["start_seconds"]))

    def test_snippets_are_word_bounded_and_capped(self):
        samples = select_speaker_samples(canonical([turn(1, "SPEAKER_00", 0, 60), turn(2, "SPEAKER_01", 60, 120)]))
        clip = samples["SPEAKER_00"][0]
        self.assertLessEqual(clip["end_seconds"] - clip["start_seconds"], 15.0)
        self.assertGreaterEqual(clip["end_seconds"] - clip["start_seconds"], 4.0)
        self.assertEqual(clip["start_seconds"], 0.0)

    def test_prefers_higher_confidence_within_a_slice(self):
        turns = [
            turn(1, "SPEAKER_00", 0, 8, confidence=0.5),
            turn(2, "SPEAKER_00", 10, 18, confidence=0.95),
            turn(3, "SPEAKER_01", 20, 28),
        ]
        picks = select_speaker_samples(canonical(turns, duration=300), per_speaker=1)["SPEAKER_00"]
        self.assertEqual(picks[0]["turn_id"], 2)

    def test_short_backchannel_turns_are_excluded(self):
        turns = [turn(1, "SPEAKER_00", 0, 8), turn(2, "SPEAKER_01", 8, 13, short=True), turn(3, "SPEAKER_01", 20, 28)]
        picks = select_speaker_samples(canonical(turns))["SPEAKER_01"]
        self.assertEqual([p["turn_id"] for p in picks], [3])

    def test_falls_back_to_shorter_turns_when_speaker_rarely_talks(self):
        turns = [turn(1, "SPEAKER_00", 0, 100), turn(2, "SPEAKER_01", 100, 102), turn(3, "SPEAKER_01", 150, 151.5), turn(4, "SPEAKER_00", 160, 200)]
        picks = select_speaker_samples(canonical(turns))["SPEAKER_01"]
        self.assertEqual(sorted(p["turn_id"] for p in picks), [2, 3])

    def test_speaker_with_no_usable_turns_gets_empty_list(self):
        turns = [turn(1, "SPEAKER_00", 0, 50), turn(2, "SPEAKER_01", 50, 50.4)]
        self.assertEqual(select_speaker_samples(canonical(turns))["SPEAKER_01"], [])

    def test_speaking_time(self):
        turns = [turn(1, "SPEAKER_00", 0, 30), turn(2, "SPEAKER_01", 30, 40), turn(3, "SPEAKER_00", 40, 45)]
        self.assertEqual(speaking_time(canonical(turns)), {"SPEAKER_00": 35.0, "SPEAKER_01": 10.0})


class TestExtractClip(unittest.TestCase):

    def test_extracts_padded_wav_within_bounds(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.wav")
            sf.write(path, np.zeros(16000 * 10, dtype=np.float32), 16000, subtype="PCM_16")
            data, sr = sf.read(io.BytesIO(extract_clip(path, 2.0, 4.0)))
            self.assertEqual(sr, 16000)
            self.assertAlmostEqual(len(data) / sr, 2.3, places=2)

            edge, _ = sf.read(io.BytesIO(extract_clip(path, 0.0, 10.0)))
            self.assertAlmostEqual(len(edge) / 16000, 10.0, places=2)


if __name__ == "__main__":
    unittest.main()
