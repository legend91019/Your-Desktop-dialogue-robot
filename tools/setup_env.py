import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VENV_DIR = PROJECT_ROOT / ".venv"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
SUPPORTED_MINORS = {(3, 10), (3, 11)}
BOOTSTRAP_SCRIPT = PROJECT_ROOT / "tools" / "bootstrap_release.ps1"


def run(command):
    print("+ " + " ".join(str(part) for part in command), flush=True)
    subprocess.check_call([str(part) for part in command], cwd=PROJECT_ROOT)


def command_output(command):
    try:
        return subprocess.check_output(
            [str(part) for part in command],
            cwd=PROJECT_ROOT,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def python_version(command):
    script = "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    output = command_output([*command, "-c", script])
    if not output:
        return None

    version_text = output.splitlines()[-1].strip()
    parts = version_text.split(".")
    if len(parts) < 2:
        return None

    try:
        return tuple(int(part) for part in parts[:3])
    except ValueError:
        return None


def is_supported(version):
    return version is not None and version[:2] in SUPPORTED_MINORS


def candidate_pythons():
    env_python = os.environ.get("XINBAO_PYTHON")
    if env_python:
        yield [env_python]

    if shutil.which("py"):
        yield ["py", "-3.11"]
        yield ["py", "-3.10"]

    if shutil.which("python"):
        yield ["python"]


def find_base_python():
    checked = []
    for command in candidate_pythons():
        version = python_version(command)
        if version:
            checked.append((command, version))
        if is_supported(version):
            print(f"Using Python {version[0]}.{version[1]}.{version[2]} from: {' '.join(command)}")
            return command

    details = "\n".join(
        f"  - {' '.join(command)} -> {version[0]}.{version[1]}.{version[2]}"
        for command, version in checked
    )
    if not details:
        details = "  - No Python launcher was found."

    raise RuntimeError(
        "Xinbao release installer requires Python 3.10 or 3.11.\n"
        "Detected candidates:\n"
        f"{details}\n\n"
        "The installer will try to create a project-local Python 3.11 runtime next."
    )


def create_venv_with_uv():
    if not BOOTSTRAP_SCRIPT.exists():
        raise RuntimeError(f"Missing bootstrap script: {BOOTSTRAP_SCRIPT}")

    powershell = shutil.which("powershell") or shutil.which("powershell.exe")
    if not powershell:
        raise RuntimeError(
            "No compatible Python was found, and PowerShell is unavailable for automatic bootstrap."
        )

    print("No compatible Python 3.10/3.11 was found.")
    print("Creating a project-local Python 3.11 runtime with uv...")
    run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            BOOTSTRAP_SCRIPT,
            "-CreateVenvOnly",
        ]
    )

    python_exe = venv_python()
    if not python_exe.exists():
        raise RuntimeError(f"uv bootstrap finished but .venv Python was not found: {python_exe}")

    return python_exe


def venv_python():
    return VENV_DIR / "Scripts" / "python.exe"


def ensure_venv():
    python_exe = venv_python()
    if python_exe.exists():
        version = python_version([python_exe])
        if is_supported(version):
            print(f"Reusing existing .venv Python {version[0]}.{version[1]}.{version[2]}")
            return python_exe

        raise RuntimeError(
            f"Existing .venv uses unsupported Python version: {version}.\n"
            "Please delete the project .venv folder and rerun install.bat."
        )

    try:
        base_python = find_base_python()
    except RuntimeError as exc:
        print(exc)
        return create_venv_with_uv()

    run([*base_python, "-m", "venv", str(VENV_DIR)])
    return python_exe


def upgrade_packaging_tools(python_exe):
    run([python_exe, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])


def install_requirements(python_exe):
    run([python_exe, "-m", "pip", "install", "-r", REQUIREMENTS_FILE])


def install_torch(python_exe, variant):
    if variant == "skip":
        print("Skipping PyTorch installation because --torch skip was requested.")
        return

    index_by_variant = {
        "cu128": "https://download.pytorch.org/whl/cu128",
        "cpu": "https://download.pytorch.org/whl/cpu",
    }
    run(
        [
            python_exe,
            "-m",
            "pip",
            "install",
            "torch",
            "torchvision",
            "torchaudio",
            "--index-url",
            index_by_variant[variant],
        ]
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create Xinbao .venv and install dependencies.")
    parser.add_argument(
        "--torch",
        choices=["cu128", "cpu", "skip"],
        default=os.environ.get("XINBAO_TORCH_VARIANT", "cu128"),
        help="PyTorch wheel variant. Default: cu128. Use cpu for non-NVIDIA machines, skip if already installed.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    python_exe = ensure_venv()
    upgrade_packaging_tools(python_exe)
    install_torch(python_exe, args.torch)
    install_requirements(python_exe)
    print("\nXinbao environment is ready.")
    print(f"Python: {python_exe}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
