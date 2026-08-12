# Release Checklist

Use this checklist before publishing a GitHub release.

## Required checks

- [ ] `README.md` states the current version.
- [ ] `config.example.json` contains no real API key.
- [ ] `config.json` in the public repository contains no real API key.
- [ ] Default `voice_settings.engine` is `edge_tts`.
- [ ] Index-TTS is documented as optional.
- [ ] `download_all_models.py` does not download Index-TTS by default.
- [ ] `install.bat` exists.
- [ ] `download_models.bat` exists.
- [ ] `start_xinbao_desktop.bat` exists.
- [ ] `docs/RELEASE_NOTES_v1.0.0.md` exists.
- [ ] No model weights are tracked.
- [ ] No `chroma_db/` data is tracked.
- [ ] No generated `static/reply_*.mp3` or `.wav` files are tracked.
- [ ] No voice recordings are tracked.
- [ ] No `__pycache__` or `.pyc` files are tracked.
- [ ] No private report documents are tracked.
- [ ] Software-side tests pass.

## Recommended commands

```powershell
python -B -m unittest discover -s tests -p "test_*.py" -v
```

```powershell
git status --short
```

```powershell
git ls-files | Select-String -Pattern "models|chroma_db|voice_records|reply_.*\.(mp3|wav)|__pycache__|\.pyc|\.env"
```

## Release artifact policy

For v1.0.0, publish the source code and Git tag first. Do not attach local models, local databases, private configs, or generated runtime files.

Future Windows installer work can be released as a later version after local testing confirms the dependency and model download flow.
