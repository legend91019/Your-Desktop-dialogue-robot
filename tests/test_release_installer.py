import unittest
from pathlib import Path
from unittest.mock import patch

from tools import setup_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ReleaseInstallerTest(unittest.TestCase):
    def test_install_bat_bootstraps_project_venv(self):
        install_bat = PROJECT_ROOT / "install.bat"

        self.assertTrue(install_bat.exists(), "install.bat should exist for release users")
        content = install_bat.read_text(encoding="utf-8")

        self.assertIn("tools\\setup_env.py", content)
        self.assertIn(".venv", content)
        self.assertIn("BOOTSTRAP_PYTHON=py", content)

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

    def test_download_models_bat_trains_classifier_for_three_step_release_flow(self):
        content = (PROJECT_ROOT / "download_models.bat").read_text(encoding="utf-8")

        self.assertIn("download_all_models.py", content)
        self.assertIn("train_classifier.py", content)

    def test_readme_does_not_contain_generation_artifacts(self):
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("markdown_content =", readme)
        self.assertNotIn("SUCCESS: README.md generated.", readme)
        self.assertIn("install.bat", readme)
        self.assertIn("download_models.bat", readme)
        self.assertIn("start_xinbao_desktop.bat", readme)

    def test_setup_env_rejects_python_313_candidates(self):
        with patch("tools.setup_env.candidate_pythons", return_value=iter([["python"]])):
            with patch("tools.setup_env.python_version", return_value=(3, 13, 5)):
                with self.assertRaisesRegex(RuntimeError, "requires Python 3.10 or 3.11"):
                    setup_env.find_base_python()

    def test_setup_env_accepts_python_311_candidate(self):
        with patch("tools.setup_env.candidate_pythons", return_value=iter([["py", "-3.11"]])):
            with patch("tools.setup_env.python_version", return_value=(3, 11, 9)):
                self.assertEqual(setup_env.find_base_python(), ["py", "-3.11"])

    def test_setup_env_defaults_to_cuda_128_torch_variant(self):
        args = setup_env.parse_args([])

        self.assertEqual(args.torch, "cu128")

    def test_setup_env_install_order(self):
        calls = []

        with patch("tools.setup_env.ensure_venv", return_value=Path(".venv/Scripts/python.exe")):
            with patch("tools.setup_env.upgrade_packaging_tools", side_effect=lambda *_: calls.append("upgrade")):
                with patch("tools.setup_env.install_torch", side_effect=lambda *_: calls.append("torch")):
                    with patch("tools.setup_env.install_requirements", side_effect=lambda *_: calls.append("requirements")):
                        setup_env.main([])

        self.assertEqual(calls, ["upgrade", "torch", "requirements"])


if __name__ == "__main__":
    unittest.main()
