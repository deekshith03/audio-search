import unittest

import numpy as np

from scripts.build_dataset import load_sources, mix_at_snr, rms, verify


class TestMixAtSnr(unittest.TestCase):

    def test_achieves_target_snr(self):
        rng = np.random.default_rng(0)
        speech = 0.1 * np.sin(np.linspace(0, 2000 * np.pi, 16000)).astype(np.float32)
        noise = rng.normal(0, 0.3, 16000).astype(np.float32)
        mixed = mix_at_snr(speech, noise, 12.0)
        snr = 20 * np.log10(rms(speech) / rms(mixed - speech))
        self.assertAlmostEqual(snr, 12.0, places=2)

    def test_short_noise_is_looped_to_speech_length(self):
        mixed = mix_at_snr(np.full(100, 0.1, dtype=np.float32), np.array([0.5, -0.5], dtype=np.float32), 10.0)
        self.assertEqual(len(mixed), 100)

    def test_output_is_peak_limited(self):
        mixed = mix_at_snr(np.full(10, 0.99, dtype=np.float32), np.full(10, 1.0, dtype=np.float32), -20.0)
        self.assertLessEqual(float(np.max(np.abs(mixed))), 1.0)

    def test_silent_noise_is_rejected(self):
        with self.assertRaises(ValueError):
            mix_at_snr(np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float32), 12.0)


class TestCommittedDataset(unittest.TestCase):

    def test_committed_audio_matches_recorded_provenance(self):
        self.assertEqual(verify(load_sources()["files"]), 0)

    def test_every_ground_truth_file_has_a_sources_entry(self):
        import os
        gt = {f.replace(".json", ".wav") for f in os.listdir("dataset/ground_truth")}
        self.assertEqual({e["file_id"] for e in load_sources()["files"]}, gt)


if __name__ == "__main__":
    unittest.main()
