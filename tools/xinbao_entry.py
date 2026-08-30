import subprocess
import sys
import os
from pathlib import Path


def main() -> int:
    root = Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent
    python_exe = root / "python" / "python.exe"
    if not python_exe.exists():
        python_exe = root / ".venv" / "Scripts" / "python.exe"
    launcher = root / "desktop_launcher.py"
    if not python_exe.exists() or not launcher.exists():
        print("芯宝运行文件不完整，请重新安装。", file=sys.stderr)
        return 1
    env = os.environ.copy()
    site_packages = root / ".venv" / "Lib" / "site-packages"
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(site_packages), env.get("PYTHONPATH", "")]))
    process = subprocess.run([str(python_exe), str(launcher)], cwd=str(root), env=env)
    return int(process.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
