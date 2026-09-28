import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.pipeline.common import (
    build_cache_key,
    file_base,
    get_hf_token,
    is_cache_valid,
    parse_stage_args,
    resolve_audio_files,
    write_json,
)


class TestCacheKeys(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.input_path = os.path.join(self.tmp.name, "input.wav")
        with open(self.input_path, "wb") as f:
            f.write(b"audio-bytes-v1")

    def tearDown(self):
        self.tmp.cleanup()

    def test_same_config_and_input_gives_same_key(self):
        self.assertEqual(
            build_cache_key({"model": "a"}, [self.input_path]),
            build_cache_key({"model": "a"}, [self.input_path]),
        )

    def test_config_change_changes_key(self):
        self.assertNotEqual(
            build_cache_key({"model": "a"}, [self.input_path]),
            build_cache_key({"model": "b"}, [self.input_path]),
        )

    def test_input_content_change_changes_key(self):
        before = build_cache_key({"model": "a"}, [self.input_path])
        with open(self.input_path, "wb") as f:
            f.write(b"audio-bytes-v2")
        self.assertNotEqual(before, build_cache_key({"model": "a"}, [self.input_path]))

    def test_key_is_independent_of_config_key_order(self):
        self.assertEqual(
            build_cache_key({"a": 1, "b": 2}, [self.input_path]),
            build_cache_key({"b": 2, "a": 1}, [self.input_path]),
        )

    def test_cache_valid_only_when_key_matches(self):
        artifact = os.path.join(self.tmp.name, "artifact.json")
        write_json(artifact, {"cache_key": "abc"})
        self.assertTrue(is_cache_valid(artifact, "abc"))
        self.assertFalse(is_cache_valid(artifact, "xyz"))

    def test_cache_invalid_when_missing_or_corrupt(self):
        missing = os.path.join(self.tmp.name, "missing.json")
        self.assertFalse(is_cache_valid(missing, "abc"))

        corrupt = os.path.join(self.tmp.name, "corrupt.json")
        with open(corrupt, "w") as f:
            f.write("{not json")
        self.assertFalse(is_cache_valid(corrupt, "abc"))

    def test_legacy_artifact_without_cache_key_is_invalid(self):
        legacy = os.path.join(self.tmp.name, "legacy.json")
        write_json(legacy, {"segments": []})
        self.assertFalse(is_cache_valid(legacy, "abc"))


class TestStageCli(unittest.TestCase):

    def test_defaults_to_all_files_without_force(self):
        args = parse_stage_args("x", [])
        self.assertIsNone(args.files)
        self.assertFalse(args.force)

    def test_repeatable_file_and_force(self):
        args = parse_stage_args("x", ["--file", "a.wav", "--file", "b.wav", "--force"])
        self.assertEqual(args.files, ["a.wav", "b.wav"])
        self.assertTrue(args.force)

    def test_resolve_lists_wavs_in_audio_dir_sorted(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ("b.wav", "a.wav", "notes.txt"):
                open(os.path.join(d, name), "w").close()
            args = parse_stage_args("x", ["--audio-dir", d])
            self.assertEqual([os.path.basename(p) for p in resolve_audio_files(args)], ["a.wav", "b.wav"])

    def test_resolve_raises_for_missing_file(self):
        args = parse_stage_args("x", ["--file", "/nonexistent/x.wav"])
        with self.assertRaises(FileNotFoundError):
            resolve_audio_files(args)

    def test_file_base_strips_directory_and_extension(self):
        self.assertEqual(file_base("dataset/audio/audio_01_x.wav"), "audio_01_x")


class TestWriteJson(unittest.TestCase):

    def test_writes_json_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "out.json")
            write_json(path, {"k": "é"})
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"k": "é"})
            self.assertEqual(os.listdir(os.path.dirname(path)), ["out.json"])


class TestHfToken(unittest.TestCase):

    def test_reads_token_from_environment(self):
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test_token"}, clear=True):
            self.assertEqual(get_hf_token(env_path=None), "hf_test_token")

    def test_reads_alternate_env_var(self):
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_alt"}, clear=True):
            self.assertEqual(get_hf_token(env_path=None), "hf_alt")

    def test_reads_token_from_env_file_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as d:
            env_path = os.path.join(d, ".env")
            with open(env_path, "w") as f:
                f.write("# comment\nHF_TOKEN='hf_from_file'\n")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(get_hf_token(env_path=env_path), "hf_from_file")
            with patch.dict(os.environ, {"HF_TOKEN": "hf_from_env"}, clear=True):
                self.assertEqual(get_hf_token(env_path=env_path), "hf_from_env")

    def test_reads_cached_cli_token(self):
        with tempfile.TemporaryDirectory() as d:
            token_path = os.path.join(d, "token")
            with open(token_path, "w") as f:
                f.write("hf_cached\n")
            with patch.dict(os.environ, {}, clear=True), patch("os.path.expanduser", return_value=token_path):
                self.assertEqual(get_hf_token(env_path=None), "hf_cached")

    def test_fails_closed_when_no_token_anywhere(self):
        with patch.dict(os.environ, {}, clear=True), patch("os.path.expanduser", return_value="/nonexistent/token"):
            with self.assertRaises(RuntimeError) as cm:
                get_hf_token(env_path="/nonexistent/.env")
            self.assertIn("Hugging Face token not found", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
