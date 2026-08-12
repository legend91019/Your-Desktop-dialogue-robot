import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_MODEL = None
_MODEL_KEY = None
_RUNTIME_DIR = None


def add_indextts_source_path(source_path=None):
    candidates = []
    if source_path:
        candidates.append(Path(source_path))
    candidates.append(Path.cwd())

    for candidate in candidates:
        if (candidate / "indextts").is_dir():
            candidate_str = str(candidate.resolve())
            site_packages = candidate / ".venv" / "Lib" / "site-packages"
            if site_packages.is_dir():
                site_packages_str = str(site_packages.resolve())
                if site_packages_str not in sys.path:
                    sys.path.insert(0, site_packages_str)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            return candidate_str
    return None


def has_non_ascii(value):
    return any(ord(ch) > 127 for ch in str(value))


def warn_non_ascii_runtime_paths(source_path):
    risky_paths = [sys.executable]
    if source_path:
        risky_paths.append(source_path)
    for path in risky_paths:
        if has_non_ascii(path):
            print(
                "WARNING: Non-ASCII runtime path detected. "
                "Index-TTS dependencies such as wetext/kaldifst may fail to open FST files. "
                f"Path: {path}"
            )


def set_runtime_dir(source_path=None):
    global _RUNTIME_DIR
    base = Path(source_path) if source_path else Path.cwd()
    _RUNTIME_DIR = base / ".xinbao_runtime"
    _RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    return _RUNTIME_DIR


def get_runtime_dir():
    return _RUNTIME_DIR or set_runtime_dir()


def ensure_ascii_directory_path(path_value, name):
    path = Path(path_value)
    if not has_non_ascii(path):
        return str(path)

    runtime_dir = get_runtime_dir()
    link_path = runtime_dir / name
    if link_path.exists():
        return str(link_path)

    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to create ASCII junction for {path}: {exc}") from exc
    return str(link_path)


def ensure_ascii_file_path(path_value, name):
    path = Path(path_value)
    if not has_non_ascii(path):
        return str(path)

    runtime_dir = get_runtime_dir()
    target = runtime_dir / f"{name}{path.suffix}"
    if not target.exists() or target.stat().st_size != path.stat().st_size:
        shutil.copy2(path, target)
    return str(target)


def ascii_output_path(path_value):
    path = Path(path_value)
    if not has_non_ascii(path):
        return str(path)

    runtime_dir = get_runtime_dir()
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    return str(runtime_dir / f"output_{digest}{path.suffix}")


def prepare_runtime_payload(payload):
    runtime_payload = dict(payload)
    original_output_path = payload["output_path"]
    model_dir = ensure_ascii_directory_path(payload["model_dir"], "checkpoints")
    runtime_payload["model_dir"] = model_dir
    runtime_payload["cfg_path"] = str(Path(model_dir) / Path(payload["cfg_path"]).name)
    runtime_payload["speaker_audio"] = ensure_ascii_file_path(payload["speaker_audio"], "speaker_audio")
    runtime_payload["output_path"] = ascii_output_path(original_output_path)
    return runtime_payload, original_output_path


def load_model(payload):
    global _MODEL, _MODEL_KEY
    from indextts.infer_v2 import IndexTTS2

    cfg_path = payload["cfg_path"]
    model_dir = payload["model_dir"]
    use_fp16 = bool(payload.get("use_fp16", False))
    use_cuda_kernel = bool(payload.get("use_cuda_kernel", False))
    use_deepspeed = bool(payload.get("use_deepspeed", False))
    model_key = (cfg_path, model_dir, use_fp16, use_cuda_kernel, use_deepspeed)

    if _MODEL is not None and _MODEL_KEY == model_key:
        return _MODEL

    print(f"Loading Index-TTS model from {model_dir}")
    _MODEL = IndexTTS2(
        cfg_path=cfg_path,
        model_dir=model_dir,
        use_fp16=use_fp16,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed,
    )
    _MODEL_KEY = model_key
    print("Index-TTS model loaded")
    return _MODEL


def synthesize(payload):
    payload, original_output_path = prepare_runtime_payload(payload)
    text = payload["text"]
    output_path = payload["output_path"]
    speaker_audio = payload["speaker_audio"]
    model_dir = payload["model_dir"]
    cfg_path = payload["cfg_path"]

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not Path(cfg_path).exists():
        raise FileNotFoundError(f"cfg_path not found: {cfg_path}")
    if not Path(speaker_audio).exists():
        raise FileNotFoundError(f"speaker_audio not found: {speaker_audio}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    model = load_model(payload)
    model.infer(
        spk_audio_prompt=speaker_audio,
        text=text,
        output_path=output_path,
        verbose=False,
    )
    if Path(output_path) != Path(original_output_path):
        Path(original_output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, original_output_path)
    return {"ok": True, "output_path": original_output_path}


class IndexTTSHandler(BaseHTTPRequestHandler):
    def _send_json(self, status, body):
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"ok": True, "service": "indextts"})
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/tts":
            self._send_json(404, {"ok": False, "error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            for key in ("text", "output_path", "model_dir", "cfg_path", "speaker_audio"):
                if not payload.get(key):
                    raise ValueError(f"missing required field: {key}")
            self._send_json(200, synthesize(payload))
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": str(exc)})

    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")


def main():
    parser = argparse.ArgumentParser(description="Local Index-TTS service for Xinbao.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument(
        "--indextts-src",
        default=None,
        help="Path to the cloned index-tts source directory. Defaults to current working directory if it contains indextts/.",
    )
    args = parser.parse_args()

    source_path = add_indextts_source_path(args.indextts_src)
    if source_path:
        print(f"Using Index-TTS source path: {source_path}")
    else:
        print("Index-TTS source path not found yet; import will rely on the active Python environment.")
    runtime_dir = set_runtime_dir(source_path)
    print(f"Using ASCII runtime path: {runtime_dir}")
    warn_non_ascii_runtime_paths(source_path)

    server = ThreadingHTTPServer((args.host, args.port), IndexTTSHandler)
    print(f"Index-TTS service listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
