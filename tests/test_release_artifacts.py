import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleaseArtifactsTest(unittest.TestCase):
    def test_v1_release_scripts_exist(self):
        for relative_path in [
            "install.bat",
            "download_models.bat",
            "models_ready.bat",
            "start_xinbao_desktop.bat",
        ]:
            path = PROJECT_ROOT / relative_path
            self.assertTrue(path.exists(), f"{relative_path} should exist for v1 release")

    def test_public_config_defaults_to_edge_tts_without_api_key(self):
        for relative_path in ["config.json", "config.example.json"]:
            with self.subTest(relative_path=relative_path):
                config = json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
                self.assertEqual(config["api_settings"]["deepseek_api_key"], "")
                self.assertEqual(config["voice_settings"]["engine"], "edge_tts")
                self.assertEqual(config["voice_settings"]["fallback_engine"], "edge_tts")

    def test_readme_documents_v1_and_optional_indextts(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("v1.0.3", readme)
        self.assertIn("install.bat", readme)
        self.assertIn("download_models.bat", readme)
        self.assertIn("start_xinbao_desktop.bat", readme)
        self.assertIn("Index-TTS", readme)
        self.assertIn("--with-indextts", readme)
        self.assertIn(".python/", readme)

    def test_release_notes_and_checklist_exist(self):
        for relative_path in [
            "docs/RELEASE_NOTES_v1.0.0.md",
            "docs/RELEASE_NOTES_v1.0.2.md",
            "docs/RELEASE_NOTES_v1.0.3.md",
            "docs/RELEASE_CHECKLIST.md",
        ]:
            path = PROJECT_ROOT / relative_path
            self.assertTrue(path.exists(), f"{relative_path} should exist")
            self.assertGreater(len(path.read_text(encoding="utf-8")), 500)


if __name__ == "__main__":
    unittest.main()
