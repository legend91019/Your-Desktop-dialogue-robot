import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import download_all_models


class DownloadAllModelsTest(unittest.TestCase):
    @patch("download_all_models.modelscope_snapshot_download")
    def test_downloads_indextts_to_project_models_directory(self, modelscope_snapshot_download):
        with TemporaryDirectory() as temp_dir:
            temp_path = download_all_models.Path(temp_dir)
            with patch.object(download_all_models, "DEFAULT_INDEXTTS_MODEL_DIR", temp_path / "IndexTTS" / "checkpoints"):
                target = download_all_models.download_indextts2()

        self.assertEqual(target, temp_path / "IndexTTS" / "checkpoints")
        modelscope_snapshot_download.assert_called_once_with(
            "IndexTeam/IndexTTS-2",
            local_dir=str(target),
        )

    def test_default_speaker_audio_path_is_in_project_models_directory(self):
        self.assertEqual(
            download_all_models.DEFAULT_INDEXTTS_SPEAKER_AUDIO,
            download_all_models.PROJECT_ROOT / "models" / "IndexTTS" / "xinbao_voice.wav",
        )

    @patch("download_all_models.download_indextts2")
    @patch("download_all_models.download_embedding")
    @patch("download_all_models.download_reranker")
    def test_default_main_downloads_only_basic_models(
        self,
        download_reranker,
        download_embedding,
        download_indextts2,
    ):
        download_all_models.main([])

        download_reranker.assert_called_once_with()
        download_embedding.assert_called_once_with()
        download_indextts2.assert_not_called()

    @patch("download_all_models.download_indextts2")
    @patch("download_all_models.download_embedding")
    @patch("download_all_models.download_reranker")
    def test_with_indextts_flag_downloads_optional_tts_model(
        self,
        download_reranker,
        download_embedding,
        download_indextts2,
    ):
        download_all_models.main(["--with-indextts"])

        download_reranker.assert_called_once_with()
        download_embedding.assert_called_once_with()
        download_indextts2.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
