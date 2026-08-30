import os
import tempfile
import unittest
from pathlib import Path

import runtime_paths


class RuntimePathsTest(unittest.TestCase):
    def test_user_data_root_uses_appdata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old = os.environ.get("APPDATA")
            os.environ["APPDATA"] = temp_dir
            try:
                self.assertEqual(runtime_paths.user_data_root(), Path(temp_dir) / "Xinbao")
            finally:
                if old is None:
                    os.environ.pop("APPDATA", None)
                else:
                    os.environ["APPDATA"] = old

    def test_relative_resource_resolves_from_project_root(self):
        self.assertEqual(runtime_paths.resolve_resource("knowledge.md").name, "knowledge.md")

    def test_ensure_user_dirs_creates_runtime_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            old = os.environ.get("APPDATA")
            os.environ["APPDATA"] = temp_dir
            try:
                paths = runtime_paths.ensure_user_dirs()
                for path in paths.values():
                    self.assertTrue(path.exists(), path)
            finally:
                if old is None:
                    os.environ.pop("APPDATA", None)
                else:
                    os.environ["APPDATA"] = old


if __name__ == "__main__":
    unittest.main()
