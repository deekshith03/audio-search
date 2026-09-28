import os
import tempfile
import unittest

import numpy as np
import torch
from pyannote.core import Annotation, Segment

from src.pipeline.diarize import (
    REFINEMENT,
    annotation_to_intervals,
    diarize_config,
    refine_short_segments_with_centroids,
    write_rttm,
)
from src.pipeline.evaluate_pipeline import parse_rttm

SR = 100
VOICES = {
    1.0: np.array([1.0, 0.6]),
    2.0: np.array([0.6, 1.0]),
    3.0: np.array([0.55, 1.0]),
}


def fake_embedding_model(chunk: torch.Tensor):
    level = round(float(chunk.mean()), 1)
    return np.array([VOICES[level]])


def build_waveform(segments):
    total = int(max(end for _, end, _ in segments) * SR)
    wav = np.zeros(total, dtype=np.float32)
    for start, end, level in segments:
        wav[int(start * SR):int(end * SR)] = level
    return torch.from_numpy(wav).reshape(1, 1, -1)


def interval(speaker, start, end):
    return {"speaker": speaker, "start": start, "end": end, "duration": round(end - start, 3)}


class TestRttmIO(unittest.TestCase):

    def test_annotation_to_intervals_and_rttm_round_trip(self):
        annotation = Annotation(uri="clip")
        annotation[Segment(0.0, 2.5)] = "SPEAKER_00"
        annotation[Segment(2.5, 4.0)] = "SPEAKER_01"
        intervals = annotation_to_intervals(annotation)
        self.assertEqual(intervals, [interval("SPEAKER_00", 0.0, 2.5), interval("SPEAKER_01", 2.5, 4.0)])

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "clip.rttm")
            write_rttm(intervals, path, "clip")
            with open(path) as f:
                self.assertEqual(f.readline().strip(), "SPEAKER clip 1 0.000 2.500 <NA> <NA> SPEAKER_00 <NA> <NA>")
            parsed = parse_rttm(path, "clip")
        self.assertEqual(annotation_to_intervals(parsed), intervals)

    def test_overlapping_raw_intervals_are_preserved(self):
        annotation = Annotation(uri="clip")
        annotation[Segment(0.0, 3.0)] = "SPEAKER_00"
        annotation[Segment(2.0, 4.0)] = "SPEAKER_01"
        self.assertEqual(len(annotation_to_intervals(annotation)), 2)


class TestCentroidRefinement(unittest.TestCase):

    def test_reassigns_short_segment_closer_to_other_speaker(self):
        segments = [(0.0, 7.0, 1.0), (7.0, 14.0, 2.0), (14.0, 16.0, 3.0)]
        intervals = [interval("SPEAKER_00", 0.0, 7.0), interval("SPEAKER_01", 7.0, 14.0), interval("SPEAKER_00", 14.0, 16.0)]
        refined, telemetry = refine_short_segments_with_centroids(intervals, build_waveform(segments), SR, fake_embedding_model)

        self.assertTrue(telemetry["applied"])
        self.assertEqual(telemetry["reassigned_segments"], 1)
        self.assertGreater(telemetry["centroid_similarity"], REFINEMENT["trigger_centroid_similarity"])
        self.assertEqual([iv["speaker"] for iv in refined], ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"])

    def test_long_segments_are_never_reassigned(self):
        segments = [(0.0, 7.0, 1.0), (7.0, 14.0, 2.0), (14.0, 21.0, 3.0)]
        intervals = [interval("SPEAKER_00", 0.0, 7.0), interval("SPEAKER_01", 7.0, 14.0), interval("SPEAKER_00", 14.0, 21.0)]
        refined, telemetry = refine_short_segments_with_centroids(intervals, build_waveform(segments), SR, fake_embedding_model)
        self.assertEqual(telemetry["reassigned_segments"], 0)
        self.assertEqual(refined[2]["speaker"], "SPEAKER_00")

    def test_skipped_when_centroids_are_well_separated(self):
        VOICES[4.0] = np.array([1.0, 0.0])
        VOICES[5.0] = np.array([0.0, 1.0])
        try:
            segments = [(0.0, 7.0, 4.0), (7.0, 14.0, 5.0), (14.0, 16.0, 5.0)]
            intervals = [interval("SPEAKER_00", 0.0, 7.0), interval("SPEAKER_01", 7.0, 14.0), interval("SPEAKER_00", 14.0, 16.0)]
            refined, telemetry = refine_short_segments_with_centroids(intervals, build_waveform(segments), SR, fake_embedding_model)
        finally:
            del VOICES[4.0], VOICES[5.0]
        self.assertFalse(telemetry["applied"])
        self.assertEqual(refined, intervals)

    def test_skipped_without_long_segments_for_both_speakers(self):
        intervals = [interval("SPEAKER_00", 0.0, 7.0), interval("SPEAKER_01", 7.0, 9.0)]
        refined, telemetry = refine_short_segments_with_centroids(
            intervals, build_waveform([(0.0, 7.0, 1.0), (7.0, 9.0, 2.0)]), SR, fake_embedding_model
        )
        self.assertFalse(telemetry["applied"])
        self.assertEqual(refined, intervals)

    def test_skipped_unless_exactly_two_speakers(self):
        intervals = [interval("SPEAKER_00", 0.0, 7.0)]
        refined, telemetry = refine_short_segments_with_centroids(intervals, build_waveform([(0.0, 7.0, 1.0)]), SR, fake_embedding_model)
        self.assertFalse(telemetry["applied"])
        self.assertEqual(refined, intervals)


class TestDiarizeConfig(unittest.TestCase):

    def test_config_pins_two_speakers_on_cpu(self):
        cfg = diarize_config()
        self.assertEqual((cfg["min_speakers"], cfg["max_speakers"], cfg["device"]), (2, 2, "cpu"))
        self.assertEqual(cfg["refinement"], REFINEMENT)


if __name__ == "__main__":
    unittest.main()
