"""Generate deterministic checksums for packaged model files."""

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(root: Path, paths: list[Path]) -> dict:
    root = Path(root).resolve()
    files = []
    for candidate in paths:
        path = Path(candidate)
        if path.is_dir():
            candidates = sorted(item for item in path.rglob("*") if item.is_file())
        else:
            candidates = [path]
        for item in candidates:
            item = item.resolve()
            relative = item.relative_to(root).as_posix()
            files.append({"path": relative, "size": item.stat().st_size, "sha256": _sha256(item)})
    return {"version": 1, "files": files}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_root = root / "models"
    paths = [
        model_root / "embedding",
        model_root / "reranker",
        root / "assets" / "classifier" / "route_classifier.joblib",
        root / "knowledge.md",
    ]
    manifest = create_manifest(root, paths)
    target = model_root / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(manifest['files'])} entries to {target}")


if __name__ == "__main__":
    main()
