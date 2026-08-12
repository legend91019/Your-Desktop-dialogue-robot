# Xinbao v1.0.0 Release Notes

This is the first formal GitHub release of Xinbao's desktop software line.

## What is included

- A stable default voice path based on `edge-tts`.
- Optional Index-TTS integration for users who want higher quality local voice cloning.
- A desktop launcher based on the existing Flask backend and local WebView page.
- Long-term memory management APIs and front-end entry points.
- Basic setup scripts: `install.bat`, `download_models.bat`, `models_ready.bat`, and `start_xinbao_desktop.bat`.
- Public example configuration: `config.example.json` and a safe `config.json` with empty API key and default `edge_tts`.
- Release safety rules in `README.md`.
- Automated software-side tests under `tests/`.

## Default voice behavior

The default voice engine is `edge_tts`.

Index-TTS is not downloaded or started by default. Users who want to try it should run:

```powershell
python download_all_models.py --with-indextts
```

Then follow `docs/indextts_local_setup.md` to prepare the official Index-TTS repository, its independent environment, the local service, and the reference audio.

## What is intentionally not included

The release does not include API keys, local ChromaDB memory data, model weights, runtime audio files, voice recordings, the Index-TTS official repository, the Index-TTS virtual environment, or private report materials.

These files are intentionally excluded to keep the public repository clean and safe.

## Known limitations

- This release is a source-code release, not a signed Windows installer.
- Users still need a Python or conda environment.
- Index-TTS can be slow on limited GPU memory and remains an optional enhancement.
- The desktop launcher is a transitional desktop shell. Future releases may move more UI logic into a dedicated desktop application.

## Suggested verification after download

```powershell
install.bat
download_models.bat
start_xinbao_desktop.bat
```

For developers:

```powershell
python -B -m unittest discover -s tests -p "test_*.py" -v
```

## Upgrade notes from v0.2.0-beta

- `README.md` now describes the project as a formal `v1.0.0` release.
- Added dedicated install and model download scripts.
- Added release notes and release checklist.
- Kept Index-TTS optional rather than making it a default dependency.
- Preserved the public repository's privacy boundary.
