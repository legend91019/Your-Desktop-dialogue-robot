import queue
import unittest

from desktop_launcher import build_backend_command, build_window_url, missing_pywebview_message, wait_for_server


class DesktopLauncherTest(unittest.TestCase):
    def test_build_window_url_uses_loopback_root(self):
        self.assertEqual(build_window_url("127.0.0.1", 5000), "http://127.0.0.1:5000/")

    def test_missing_pywebview_message_tells_user_how_to_install(self):
        message = missing_pywebview_message()

        self.assertIn("pywebview", message)
        self.assertIn("pip install pywebview", message)

    def test_wait_for_server_raises_backend_startup_error_immediately(self):
        errors = queue.Queue()
        errors.put(RuntimeError("backend import failed"))

        with self.assertRaisesRegex(RuntimeError, "backend import failed"):
            wait_for_server("http://127.0.0.1:5999/", timeout_seconds=5, startup_errors=errors)

    def test_backend_command_runs_module_from_project_root(self):
        command = build_backend_command("127.0.0.1", 5999)
        self.assertIn("-m", command)
        self.assertIn("BackEnd.simple", command)
        self.assertNotIn("BackEnd\\simple.py", command)

    def test_backend_command_uses_module_entrypoint(self):
        command = build_backend_command("127.0.0.1", 5999)
        self.assertEqual(command[2], "BackEnd.simple")


if __name__ == "__main__":
    unittest.main()
