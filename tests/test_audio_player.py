import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from BackEnd.audio_player import build_windows_player_script, play_audio_file


class AudioPlayerTest(unittest.TestCase):
    def test_windows_player_script_escapes_single_quotes(self):
        script = build_windows_player_script(Path("D:/tmp/xinbao's voice.mp3"))

        self.assertIn("xinbao''s voice.mp3", script)
        self.assertIn("WMPlayer.OCX", script)
        self.assertIn("playState -ne 3", script)
        self.assertIn("播放启动超时", script)

    @patch("BackEnd.audio_player.subprocess.Popen")
    @patch("BackEnd.audio_player.platform.system", return_value="Windows")
    def test_play_audio_file_starts_hidden_powershell_on_windows(self, _system, popen):
        popen.return_value = Mock(pid=1234)

        result = play_audio_file("D:/tmp/reply.mp3")

        self.assertTrue(result["attempted"])
        self.assertFalse(result["played"])
        self.assertEqual(result["method"], "windows_media_player")
        args = popen.call_args.args[0]
        self.assertEqual(args[0], "powershell")
        self.assertIn("-STA", args)
        self.assertIn("-WindowStyle", args)
        self.assertIn("Hidden", args)

    @patch("BackEnd.audio_player.platform.system", return_value="Linux")
    def test_play_audio_file_reports_unsupported_platform(self, _system):
        result = play_audio_file("/tmp/reply.mp3")

        self.assertFalse(result["played"])
        self.assertEqual(result["method"], "unsupported")


if __name__ == "__main__":
    unittest.main()
