import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_INDEXTTS_MODEL_DIR = MODELS_DIR / "IndexTTS" / "checkpoints"
DEFAULT_INDEXTTS_SPEAKER_AUDIO = MODELS_DIR / "IndexTTS" / "xinbao_voice.wav"


def modelscope_snapshot_download(*args, **kwargs):
    from modelscope.hub.snapshot_download import snapshot_download as ms_snapshot_download

    return ms_snapshot_download(*args, **kwargs)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_reranker():
    target = ensure_dir(MODELS_DIR / "reranker")
    print(f"[1/3] Downloading bge-reranker-base to {target}")
    modelscope_snapshot_download(
        "Xorbits/bge-reranker-base",
        cache_dir=str(target),
        local_dir=str(target),
    )
    return target


def download_embedding():
    target = ensure_dir(MODELS_DIR / "embedding")
    print(f"[2/3] Downloading bge-small-zh-v1.5 to {target}")
    modelscope_snapshot_download(
        "AI-ModelScope/bge-small-zh-v1.5",
        cache_dir=str(target),
        local_dir=str(target),
    )
    return target


def download_indextts2():
    target = ensure_dir(DEFAULT_INDEXTTS_MODEL_DIR)
    print(f"[optional] Downloading IndexTTS2 from ModelScope to {target}")
    modelscope_snapshot_download(
        "IndexTeam/IndexTTS-2",
        local_dir=str(target),
    )
    print(f"Index-TTS speaker prompt should be placed at: {DEFAULT_INDEXTTS_SPEAKER_AUDIO}")
    return target


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Download Xinbao model assets.")
    parser.add_argument(
        "--with-indextts",
        action="store_true",
        help="Also download optional Index-TTS checkpoints. This is large and not needed for the default edge-tts mode.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("Starting centralized model download into project models/ directory.")
    download_reranker()
    download_embedding()
    if args.with_indextts:
        download_indextts2()
    else:
        print("Skipping optional Index-TTS checkpoints. Use --with-indextts if you need them.")
    print("Requested model assets are ready under models/. Do not commit model files to Git.")


if __name__ == "__main__":
    main()
