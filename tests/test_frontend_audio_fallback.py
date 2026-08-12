import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FrontendAudioFallbackTest(unittest.TestCase):
    def test_frontend_does_not_skip_browser_fallback_for_unverified_local_audio(self):
        html = (PROJECT_ROOT / "FrontEnd" / "robot.html").read_text(encoding="utf-8")

        self.assertNotIn("dataObj.local_audio && dataObj.local_audio.played", html)
        self.assertIn("robotVoicePlayer", html)
        self.assertIn("player.play()", html)


if __name__ == "__main__":
    unittest.main()
