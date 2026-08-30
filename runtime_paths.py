"""Path helpers for installed resources and per-user writable data."""

import os
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent


def project_root() -> Path:
    return _PROJECT_ROOT


def user_data_root() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "Xinbao"
    return Path.home() / "AppData" / "Roaming" / "Xinbao"


def config_path() -> Path:
    return user_data_root() / "config.json"


def resolve_resource(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    resolved = (_PROJECT_ROOT / path).resolve()
    try:
        resolved.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"resource path escapes project root: {value}") from exc
    return resolved


def ensure_user_dirs() -> dict[str, Path]:
    root = user_data_root()
    paths = {
        "root": root,
        "chroma_db": root / "chroma_db",
        "uploads": root / "uploads",
        "audio": root / "static" / "audio",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths
