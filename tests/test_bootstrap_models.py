import json
import os
import tempfile
import unittest
from unittest.mock import patch

import httpx
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError

from scripts import bootstrap_models


def http_error(cls, status):
    request = httpx.Request("GET", "https://huggingface.co/api/models/x")
    return cls("error", response=httpx.Response(status, request=request))


class BootstrapTestCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"MODEL_CACHE_DIR": self.tmp.name}, clear=True)
        self.env.start()
        self.no_env_file = patch.object(bootstrap_models, "load_env_file", lambda *a, **k: None)
        self.no_env_file.start()

    def tearDown(self):
        self.no_env_file.stop()
        self.env.stop()
        self.tmp.cleanup()

    def run_main(self):
        with self.assertRaises(SystemExit) as cm:
            bootstrap_models.main()
        return cm.exception.code


class TestBootstrapFailures(BootstrapTestCase):

    def test_missing_token_exits_2_with_instructions(self):
        with patch("sys.stderr") as stderr:
            self.assertEqual(self.run_main(), 2)
        message = "".join(c.args[0] for c in stderr.write.call_args_list)
        self.assertIn("HF_TOKEN is not set", message)
        self.assertIn(bootstrap_models.PYANNOTE_TERMS_URL, message)
        self.assertIn(bootstrap_models.EMBEDDING_TERMS_URL, message)

    def test_unaccepted_terms_exit_4(self):
        os.environ["HF_TOKEN"] = "hf_test"
        with patch("huggingface_hub.snapshot_download", side_effect=http_error(GatedRepoError, 403)), patch("sys.stderr"):
            self.assertEqual(self.run_main(), 4)

    def test_unaccepted_embedding_terms_exit_4_and_name_the_model(self):
        os.environ["HF_TOKEN"] = "hf_test"
        with patch.object(bootstrap_models, "download_diarization"), \
                patch.object(bootstrap_models, "download_asr"), \
                patch.object(bootstrap_models, "download_alignment"), \
                patch("huggingface_hub.snapshot_download", side_effect=http_error(GatedRepoError, 403)) as download, \
                patch("sys.stderr") as stderr:
            self.assertEqual(self.run_main(), 4)
        download.assert_called_once_with("google/embeddinggemma-300m", token="hf_test")
        self.assertIn("google/embeddinggemma-300m", "".join(c.args[0] for c in stderr.write.call_args_list))
        self.assertFalse(os.path.exists(bootstrap_models.marker_path()))

    def test_rejected_token_exits_3(self):
        os.environ["HF_TOKEN"] = "hf_bad"
        with patch("huggingface_hub.snapshot_download", side_effect=http_error(HfHubHTTPError, 401)), patch("sys.stderr"):
            self.assertEqual(self.run_main(), 3)

    def test_unauthorized_repo_exits_3(self):
        os.environ["HF_TOKEN"] = "hf_bad"
        with patch("huggingface_hub.snapshot_download", side_effect=http_error(RepositoryNotFoundError, 404)), patch("sys.stderr"):
            self.assertEqual(self.run_main(), 3)

    def test_other_download_failure_exits_1(self):
        os.environ["HF_TOKEN"] = "hf_test"
        with patch.object(bootstrap_models, "download_diarization"), \
                patch.object(bootstrap_models, "download_asr", side_effect=OSError("disk full")), \
                patch("sys.stderr"):
            self.assertEqual(self.run_main(), 1)
        self.assertFalse(os.path.exists(bootstrap_models.marker_path()))


class TestBootstrapSuccess(BootstrapTestCase):

    def test_success_writes_marker_and_second_run_skips_downloads(self):
        os.environ["HF_TOKEN"] = "hf_test"
        with patch.object(bootstrap_models, "download_diarization") as diar, \
                patch.object(bootstrap_models, "download_asr") as asr, \
                patch.object(bootstrap_models, "download_alignment") as align, \
                patch.object(bootstrap_models, "download_embedding") as embed:
            bootstrap_models.main()
        diar.assert_called_once_with("hf_test")
        asr.assert_called_once()
        align.assert_called_once()
        embed.assert_called_once_with("hf_test")
        with open(bootstrap_models.marker_path()) as f:
            self.assertEqual(json.load(f), bootstrap_models.expected_marker())

        with patch.object(bootstrap_models, "download_asr") as asr_again:
            bootstrap_models.main()
        asr_again.assert_not_called()

    def test_changed_models_invalidate_marker(self):
        with open(bootstrap_models.marker_path(), "w") as f:
            json.dump({"asr": "old-model"}, f)
        self.assertFalse(bootstrap_models.already_bootstrapped())

    def test_marker_from_before_search_embeddings_triggers_a_download(self):
        marker = {k: v for k, v in bootstrap_models.expected_marker().items() if k != "embedding"}
        with open(bootstrap_models.marker_path(), "w") as f:
            json.dump(marker, f)
        self.assertFalse(bootstrap_models.already_bootstrapped())
        self.assertEqual(bootstrap_models.expected_marker()["embedding"], "google/embeddinggemma-300m")

    def test_marker_lives_in_model_cache_dir(self):
        self.assertEqual(bootstrap_models.marker_path(), os.path.join(self.tmp.name, ".bootstrap.json"))


if __name__ == "__main__":
    unittest.main()
