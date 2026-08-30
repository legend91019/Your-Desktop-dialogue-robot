import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InstallerConfigTest(unittest.TestCase):
    def test_iss_creates_shortcuts_and_uninstall_entry(self):
        text = (PROJECT_ROOT / "packaging" / "Xinbao.iss").read_text(encoding="utf-8")
        self.assertIn("[Icons]", text)
        self.assertIn("[UninstallDelete]", text)

    def test_iss_does_not_install_hardware_product(self):
        text = (PROJECT_ROOT / "packaging" / "Xinbao.iss").read_text(encoding="utf-8")
        self.assertNotIn("hardware_product", text)


if __name__ == "__main__":
    unittest.main()
