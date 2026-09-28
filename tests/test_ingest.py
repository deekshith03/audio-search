import json
import os
import shutil
import subprocess
import tempfile
import unittest

import soundfile as sf

from src.pipeline.ingest import IngestError, format_duration, ingest, validate_upload

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_tone(path: str, seconds: float, sample_rate: int = 44100, channels: int = 2) -> str:
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate={sample_rate}:duration={seconds}",
            "-ac", str(channels), path,
        ],
        check=True,
    )
    return path


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg/ffprobe not installed")
class TestIngest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src")
        self.out = os.path.join(self.tmp.name, "out")
        os.makedirs(self.src)

    def tearDown(self):
        self.tmp.cleanup()

    def test_supported_formats_normalize_to_16k_mono_pcm16(self):
        for ext in (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus", ".webm"):
            with self.subTest(ext=ext):
                src = make_tone(os.path.join(self.src, f"tone{ext}"), 11)
                res = ingest(src, out_dir=self.out)
                info = sf.info(res.wav_path)
                self.assertEqual((info.samplerate, info.channels, info.subtype), (16000, 1, "PCM_16"))
                self.assertAlmostEqual(info.duration, 11.0, delta=0.1)
                self.assertEqual(res.source_channels, 2)

    def test_file_id_is_content_hash_and_metadata_is_written(self):
        src = make_tone(os.path.join(self.src, "tone.mp3"), 11)
        res = ingest(src, out_dir=self.out)
        self.assertEqual(res.file_id, f"upload_{res.source_sha256[:16]}.wav")
        self.assertEqual(res.wav_path, os.path.join(self.out, res.file_id))
        with open(os.path.join(self.out, f"upload_{res.source_sha256[:16]}.ingest.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["source_filename"], "tone.mp3")
        self.assertEqual(meta["source_codec"], "mp3")

    def test_reupload_of_same_bytes_reuses_existing_wav(self):
        src = make_tone(os.path.join(self.src, "tone.flac"), 11)
        first = ingest(src, out_dir=self.out)
        renamed = os.path.join(self.src, "renamed.flac")
        shutil.copy(src, renamed)
        second = ingest(renamed, out_dir=self.out)
        self.assertFalse(first.reused_existing)
        self.assertTrue(second.reused_existing)
        self.assertEqual(first.file_id, second.file_id)

    def test_explicit_file_id_is_respected(self):
        src = make_tone(os.path.join(self.src, "tone.wav"), 11)
        self.assertEqual(ingest(src, out_dir=self.out, file_id="audio_99_custom").file_id, "audio_99_custom.wav")
        self.assertEqual(ingest(src, out_dir=self.out, file_id="audio_99_custom.wav").file_id, "audio_99_custom.wav")

    def test_rejects_audio_longer_than_limit(self):
        src = make_tone(os.path.join(self.src, "long.wav"), 13)
        with self.assertRaises(IngestError) as cm:
            ingest(src, out_dir=self.out, max_duration=12)
        self.assertIn("too long", str(cm.exception))
        self.assertFalse(os.path.exists(self.out) and os.listdir(self.out))

    def test_rejects_audio_shorter_than_minimum(self):
        src = make_tone(os.path.join(self.src, "short.wav"), 3)
        with self.assertRaises(IngestError) as cm:
            validate_upload(src)
        self.assertIn("too short", str(cm.exception))

    def test_rejects_unsupported_extension(self):
        path = os.path.join(self.src, "notes.txt")
        with open(path, "w") as f:
            f.write("hello")
        with self.assertRaises(IngestError) as cm:
            validate_upload(path)
        self.assertIn("Unsupported file type", str(cm.exception))

    def test_rejects_empty_file(self):
        path = os.path.join(self.src, "empty.mp3")
        open(path, "wb").close()
        with self.assertRaises(IngestError) as cm:
            validate_upload(path)
        self.assertIn("empty", str(cm.exception))

    def test_rejects_undecodable_file(self):
        path = os.path.join(self.src, "corrupt.mp3")
        with open(path, "wb") as f:
            f.write(os.urandom(4096))
        with self.assertRaises(IngestError):
            validate_upload(path)

    def test_rejects_video_without_audio_stream(self):
        path = os.path.join(self.src, "silent_video.mp4")
        subprocess.run(
            ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=duration=11:size=64x64:rate=5",
             "-c:v", "mpeg4", path],
            check=True,
        )
        with self.assertRaises(IngestError) as cm:
            validate_upload(path)
        self.assertIn("no audio stream", str(cm.exception))

    def test_rejects_missing_file(self):
        with self.assertRaises(IngestError):
            validate_upload(os.path.join(self.src, "nope.wav"))


class TestFormatDuration(unittest.TestCase):

    def test_formats_minutes_and_seconds(self):
        self.assertEqual(format_duration(600), "10:00")
        self.assertEqual(format_duration(724.6), "12:04")


if __name__ == "__main__":
    unittest.main()
