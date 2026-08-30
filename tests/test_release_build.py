import tempfile
import unittest
from pathlib import Path

from tools.create_model_manifest import create_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleaseBuildTest(unittest.TestCase):
    def test_build_script_uses_python_311_and_venv(self):
        script = (PROJECT_ROOT / "tools" / "build_runtime.ps1").read_text(encoding="utf-8")
        self.assertIn("3.11", script)
        self.assertIn("-m venv", script)

    def test_manifest_contains_required_model_file_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "models" / "embedding" / "config.json"
            model.parent.mkdir(parents=True)
            model.write_text("{}", encoding="utf-8")
            manifest = create_manifest(root, [model])
        self.assertEqual(manifest["files"][0]["path"], "models/embedding/config.json")
        self.assertEqual(len(manifest["files"][0]["sha256"]), 64)

    def test_installer_build_forwards_python_executable(self):
        script = (PROJECT_ROOT / "tools" / "build_installer.ps1").read_text(encoding="utf-8")
        self.assertIn("PythonExe", script)
        self.assertIn("-PythonExe", script)

    def test_runtime_build_derives_base_python_path_without_stdout_parsing(self):
        script = (PROJECT_ROOT / "tools" / "build_runtime.ps1").read_text(encoding="utf-8")
        self.assertIn("Resolve-Path -LiteralPath $PythonExe", script)


if __name__ == "__main__":
    unittest.main()
