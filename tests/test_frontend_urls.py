import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontendUrlsTest(unittest.TestCase):
    def test_frontend_has_no_hardcoded_backend_origin(self):
        html = (PROJECT_ROOT / "FrontEnd" / "robot.html").read_text(encoding="utf-8")
        self.assertNotIn("http://127.0.0.1:5000", html)

    def test_frontend_uses_relative_api_paths(self):
        html = (PROJECT_ROOT / "FrontEnd" / "robot.html").read_text(encoding="utf-8")
        self.assertIn("fetch('/api/chat'", html)
        self.assertIn("fetch('/api/memories'", html)

    def test_settings_status_never_returns_api_key(self):
        source = (PROJECT_ROOT / "BackEnd" / "simple.py").read_text(encoding="utf-8")
        self.assertIn("/api/settings/status", source)
        self.assertIn("has_api_key", source)


if __name__ == "__main__":
    unittest.main()
