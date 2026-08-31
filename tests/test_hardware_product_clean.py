import json
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HARDWARE = ROOT / "hardware_product"


class HardwareProductCleanTest(unittest.TestCase):
    def test_public_config_is_neutral_and_secret_free(self):
        config = json.loads((HARDWARE / "config.example.json").read_text(encoding="utf-8"))

        self.assertEqual(config["api_settings"]["deepseek_api_key"], "")
        self.assertEqual(config["user_settings"]["master_name"], "主人")
        self.assertEqual(config["user_settings"]["occupation"], "未设置")
        self.assertEqual(config["user_settings"]["current_status"], "未设置")

    def test_runtime_config_is_not_tracked(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "hardware_product/config.json"],
            cwd=ROOT,
            text=True,
        ).strip()
        self.assertEqual(tracked, "")
        self.assertIn("hardware_product/config.json", (ROOT / ".gitignore").read_text(encoding="utf-8"))

    def test_setup_creates_runtime_config_and_downloads_models(self):
        setup = (HARDWARE / "setup_pi.sh").read_text(encoding="utf-8")

        self.assertIn("config.example.json", setup)
        self.assertIn("snapshot_download", setup)
        self.assertIn('MODELS_DIR="$SCRIPT_DIR/models"', setup)

    def test_repository_has_voice_loop_but_no_downloaded_models(self):
        self.assertTrue((HARDWARE / "m260c_voice_loop.py").is_file())
        self.assertTrue((HARDWARE / "run_voice_loop.sh").is_file())
        self.assertFalse((HARDWARE / "models").exists())
        self.assertEqual((HARDWARE / "dynamic_keywords.txt").read_text(encoding="utf-8").strip(), "")


if __name__ == "__main__":
    unittest.main()
