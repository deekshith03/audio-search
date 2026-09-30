import json
import os
import tempfile
import threading
import unittest
from unittest.mock import patch

from src.pipeline.common import (
    GOLDEN,
    Workspace,
    build_cache_key,
    file_base,
    get_hf_token,
    hf_token_path,
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


class TestWorkspace(unittest.TestCase):

    def test_golden_workspace_matches_dataset_layout(self):
        self.assertEqual(GOLDEN.audio_dir, "dataset/audio")
        self.assertEqual(GOLDEN.raw_asr_dir, "dataset/cache/raw_asr")
        self.assertEqual(GOLDEN.diarization_dir, "dataset/cache/raw_diarization")
        self.assertEqual(GOLDEN.canonical_path("audio_01"), "dataset/pipeline_outputs/audio_01_canonical.json")

    def test_upload_workspace_uses_same_layout(self):
        ws = Workspace("data")
        self.assertEqual(
            (ws.labels_dir, ws.jobs_dir, ws.uploads_dir),
            ("data/speaker_labels", "data/jobs", "data/uploads"),
        )

    def test_cli_workspace_sets_audio_dir(self):
        args = parse_stage_args("x", ["--workspace", "data"])
        self.assertEqual(args.workspace, Workspace("data"))
        self.assertEqual(args.audio_dir, "data/audio")

    def test_explicit_audio_dir_overrides_workspace(self):
        self.assertEqual(parse_stage_args("x", ["--workspace", "data", "--audio-dir", "x"]).audio_dir, "x")

    def test_root_is_normalized_so_it_is_one_workspace_key(self):
        for spelled in ("dataset/", "./dataset", "dataset//", "dataset/../dataset"):
            self.assertEqual(Workspace(spelled).root, "dataset", spelled)
        self.assertEqual(Workspace("/app/data/").root, "/app/data")
        self.assertEqual(Workspace("dataset/"), GOLDEN)


class TestWriteJson(unittest.TestCase):

    def test_writes_json_and_leaves_no_temp_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "out.json")
            write_json(path, {"k": "é"})
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), {"k": "é"})
            self.assertEqual(os.listdir(os.path.dirname(path)), ["out.json"])

    def test_concurrent_writers_never_collide_or_leave_temp_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "job.json")
            errors = []

            def write(n):
                try:
                    for i in range(50):
                        write_json(path, {"writer": n, "i": i})
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            with open(path) as f:
                self.assertEqual(json.load(f)["i"], 49)
            self.assertEqual(os.listdir(tmp), ["job.json"])

    def test_failed_write_keeps_the_old_file_and_no_temp_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "job.json")
            write_json(path, {"ok": 1})
            with self.assertRaises(TypeError):
                write_json(path, {"bad": object()})
            with open(path) as f:
                self.assertEqual(json.load(f), {"ok": 1})
            self.assertEqual(os.listdir(tmp), ["job.json"])


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
            with patch.dict(os.environ, {}, clear=True), patch("os.path.expanduser", return_value=d):
                self.assertEqual(get_hf_token(env_path=None), "hf_cached")

    def test_reads_token_saved_under_hf_home_as_in_docker(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "token"), "w") as f:
                f.write("hf_in_hf_home\n")
            with patch.dict(os.environ, {"HF_HOME": d}, clear=True):
                self.assertEqual(hf_token_path(), os.path.join(d, "token"))
                self.assertEqual(get_hf_token(env_path=None), "hf_in_hf_home")

    def test_hf_token_path_prefers_explicit_path(self):
        with patch.dict(os.environ, {"HF_TOKEN_PATH": "/x/tok", "HF_HOME": "/y"}, clear=True):
            self.assertEqual(hf_token_path(), "/x/tok")

    def test_fails_closed_when_no_token_anywhere(self):
        with patch.dict(os.environ, {}, clear=True), patch("os.path.expanduser", return_value="/nonexistent"):
            with self.assertRaises(RuntimeError) as cm:
                get_hf_token(env_path="/nonexistent/.env")
            self.assertIn("Hugging Face token not found", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
