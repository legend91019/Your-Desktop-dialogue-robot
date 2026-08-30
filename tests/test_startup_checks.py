import socket
import tempfile
import unittest
from pathlib import Path

import startup_checks


class StartupChecksTest(unittest.TestCase):
    def test_find_free_port_returns_bindable_port(self):
        port = startup_checks.find_free_port()
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))

    def test_model_check_reports_missing_embedding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = startup_checks.check_models(Path(temp_dir))
        self.assertFalse(result.ok)
        self.assertIn("embedding", result.message)

    def test_check_result_is_structured(self):
        result = startup_checks.CheckResult(True, "ok", "ready", {"value": 1})
        self.assertTrue(result.ok)
        self.assertEqual(result.code, "ok")
        self.assertEqual(result.details["value"], 1)


if __name__ == "__main__":
    unittest.main()
