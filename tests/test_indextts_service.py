import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.indextts_service import add_indextts_source_path, prepare_runtime_payload, set_runtime_dir


class IndexTTSServiceTest(unittest.TestCase):
    def test_adds_source_path_when_directory_contains_indextts_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            (source / "indextts").mkdir()
            site_packages = source / ".venv" / "Lib" / "site-packages"
            site_packages.mkdir(parents=True)

            added = add_indextts_source_path(str(source))

            self.assertEqual(added, str(source.resolve()))
            self.assertEqual(sys.path[0], str(source.resolve()))
            self.assertEqual(sys.path[1], str(site_packages.resolve()))
            sys.path.remove(str(source.resolve()))
            sys.path.remove(str(site_packages.resolve()))

    def test_returns_none_when_source_path_does_not_contain_indextts_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(add_indextts_source_path(tmp))

    @patch("tools.indextts_service.subprocess.run")
    def test_prepare_runtime_payload_maps_non_ascii_paths_to_runtime_directory(self, subprocess_run):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_base = Path(tmp) / "ascii-src"
            runtime_base.mkdir()
            set_runtime_dir(str(runtime_base))

            source_root = Path(tmp) / "中文项目"
            model_dir = source_root / "models" / "IndexTTS" / "checkpoints"
            model_dir.mkdir(parents=True)
            (model_dir / "config.yaml").write_text("config", encoding="utf-8")
            speaker = source_root / "models" / "IndexTTS" / "xinbao_voice.wav"
            speaker.write_bytes(b"fake wav")
            output = source_root / "static" / "reply.wav"

            payload, original_output = prepare_runtime_payload(
                {
                    "text": "hello",
                    "output_path": str(output),
                    "model_dir": str(model_dir),
                    "cfg_path": str(model_dir / "config.yaml"),
                    "speaker_audio": str(speaker),
                }
            )

            self.assertEqual(original_output, str(output))
            self.assertEqual(Path(payload["model_dir"]), runtime_base / ".xinbao_runtime" / "checkpoints")
            self.assertEqual(Path(payload["cfg_path"]), runtime_base / ".xinbao_runtime" / "checkpoints" / "config.yaml")
            self.assertEqual(Path(payload["speaker_audio"]), runtime_base / ".xinbao_runtime" / "speaker_audio.wav")
            self.assertTrue(Path(payload["speaker_audio"]).exists())
            self.assertEqual(Path(payload["output_path"]).parent, runtime_base / ".xinbao_runtime")
            subprocess_run.assert_called_once()


if __name__ == "__main__":
    unittest.main()
