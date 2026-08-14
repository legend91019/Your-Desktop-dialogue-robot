import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from unittest.mock import patch

from tools import setup_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleaseInstallerTest(unittest.TestCase):
    def test_install_bat_bootstraps_project_venv(self):
        install_bat = PROJECT_ROOT / "install.bat"

        self.assertTrue(install_bat.exists(), "install.bat should exist for release users")
        content = install_bat.read_text(encoding="utf-8")

        self.assertIn("tools\\bootstrap_release.ps1", content)
        self.assertIn(".venv", content)
        self.assertIn("powershell", content.lower())

    def test_release_has_powershell_bootstrap_for_missing_python_311(self):
        bootstrap = PROJECT_ROOT / "tools" / "bootstrap_release.ps1"

        self.assertTrue(bootstrap.exists(), "release should bootstrap Python without manual env creation")
        content = bootstrap.read_text(encoding="utf-8")

        self.assertIn("uv venv", content)
        self.assertIn("--python 3.11", content)
        self.assertIn("UV_PYTHON_INSTALL_DIR", content)
        self.assertIn("--seed", content)
        self.assertIn(".python", content)
        self.assertIn(".tools", content)

    def test_requirements_do_not_pin_cuda_torch_wheels(self):
        requirements = (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8")

        self.assertNotIn("+cu128", requirements)
        self.assertNotRegex(requirements, r"(?m)^torch(?:vision|audio)?==")

    def test_runtime_bats_use_project_venv(self):
        for filename in ["download_models.bat", "models_ready.bat", "start_xinbao_desktop.bat"]:
            with self.subTest(filename=filename):
                script = PROJECT_ROOT / filename
                self.assertTrue(script.exists(), f"{filename} should exist")
                content = script.read_text(encoding="utf-8")
                self.assertIn(".venv\\Scripts\\python.exe", content)

    def test_download_models_bat_does_not_train_classifier_for_release_flow(self):
        for filename in ["download_models.bat", "models_ready.bat"]:
            with self.subTest(filename=filename):
                content = (PROJECT_ROOT / filename).read_text(encoding="utf-8")

                self.assertIn("download_all_models.py", content)
                self.assertNotIn("train_classifier.py", content)
                self.assertIn("route_classifier.joblib", content)

    def test_release_ships_lightweight_route_classifier_artifact(self):
        artifact = PROJECT_ROOT / "assets" / "classifier" / "route_classifier.joblib"

        self.assertTrue(artifact.exists(), "release should include a ready-to-use lightweight route classifier")
        self.assertLess(artifact.stat().st_size, 5 * 1024 * 1024)

    def test_backend_uses_lightweight_route_classifier(self):
        backend = (PROJECT_ROOT / "BackEnd" / "simple.py").read_text(encoding="utf-8")

        self.assertIn("load_route_classifier", backend)
        self.assertIn("route_classifier.joblib", backend)
        self.assertNotIn("TextClassifier", backend)

    def test_readme_does_not_contain_generation_artifacts(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("markdown_content =", readme)
        self.assertNotIn("SUCCESS: README.md generated.", readme)
        self.assertIn("install.bat", readme)
        self.assertIn("download_models.bat", readme)
        self.assertIn("start_xinbao_desktop.bat", readme)

    def test_setup_env_falls_back_to_managed_python_when_candidates_are_unsupported(self):
        with TemporaryDirectory() as temp_dir:
            temp_venv = Path(temp_dir) / ".venv"
            expected_python = temp_venv / "Scripts" / "python.exe"
            with patch("tools.setup_env.VENV_DIR", temp_venv):
                with patch("tools.setup_env.candidate_pythons", return_value=iter([["python"]])):
                    with patch("tools.setup_env.python_version", return_value=(3, 13, 5)):
                        with patch("tools.setup_env.create_venv_with_uv", return_value=expected_python) as managed:
                            self.assertEqual(setup_env.ensure_venv(), expected_python)

        managed.assert_called_once()

    def test_setup_env_accepts_python_311_candidate(self):
        with patch("tools.setup_env.candidate_pythons", return_value=iter([["py", "-3.11"]])):
            with patch("tools.setup_env.python_version", return_value=(3, 11, 9)):
                self.assertEqual(setup_env.find_base_python(), ["py", "-3.11"])

    def test_setup_env_defaults_to_cpu_torch_variant(self):
        args = setup_env.parse_args([])

        self.assertEqual(args.torch, "cpu")

    def test_setup_env_install_order(self):
        calls = []

        with patch("tools.setup_env.ensure_venv", return_value=Path(".venv/Scripts/python.exe")):
            with patch("tools.setup_env.upgrade_packaging_tools", side_effect=lambda *_: calls.append("upgrade")):
                with patch("tools.setup_env.install_torch", side_effect=lambda *_: calls.append("torch")):
                    with patch("tools.setup_env.install_requirements", side_effect=lambda *_: calls.append("requirements")):
                        setup_env.main([])

        self.assertEqual(calls, ["upgrade", "torch", "requirements"])

    def test_gitignore_excludes_release_bootstrap_artifacts(self):
        gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

        self.assertIn(".python/", gitignore)
        self.assertIn(".tools/", gitignore)


if __name__ == "__main__":
    unittest.main()
