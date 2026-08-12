import asyncio
import re
import unicodedata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEXTTS_MODEL_DIR = PROJECT_ROOT / "models" / "IndexTTS" / "checkpoints"
_TTS_ALLOWED_PUNCTUATION = set("，。！？；：、,.!?;: “”“”‘’'\"…-—")
_TTS_EMOTICON_PATTERN = re.compile(
    r"(?i)\b(?:qaq|qwq|tat|orz|www+)\b|"
    r"[\(（][^\u4e00-\u9fffA-Za-z0-9]{1,20}[\)）]"
)


def get_default_indextts_model_dir():
    return str(DEFAULT_INDEXTTS_MODEL_DIR)


def get_default_indextts_cfg_path():
    return str(DEFAULT_INDEXTTS_MODEL_DIR / "config.yaml")


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(PROJECT_ROOT / path)


def get_tts_extension(voice_config):
    engine = (voice_config or {}).get("engine", "edge_tts")
    if str(engine).lower() == "indextts":
        return ".wav"
    return ".mp3"


def sanitize_tts_text(text):
    text = str(text or "")
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)|（.*?）", "", text)
    text = _TTS_EMOTICON_PATTERN.sub("", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[*#`~_^|<>{}\\/\[\]=+@]+", " ", text)

    cleaned = []
    for ch in text:
        category = unicodedata.category(ch)
        if "\u4e00" <= ch <= "\u9fff":
            cleaned.append(ch)
        elif ch.isascii() and ch.isalnum():
            cleaned.append(ch)
        elif ch in _TTS_ALLOWED_PUNCTUATION:
            cleaned.append(ch)
        elif category.startswith("Z"):
            cleaned.append(" ")

    result = "".join(cleaned)
    result = re.sub(r"\s+", " ", result)
    result = re.sub(r"([，。！？；：、,.!?;:])\1+", r"\1", result)
    result = result.strip(" ，。！？；：、,.!?;:;-—")
    return result


def limit_tts_text(text, max_chars=60, max_sentences=1):
    text = str(text or "").strip()
    if not text:
        return ""

    sentence_parts = re.findall(r"[^。！？.!?]+[。！？.!?]?", text)
    if sentence_parts:
        limited = "".join(sentence_parts[:max(1, int(max_sentences))]).strip()
    else:
        limited = text

    max_chars = max(1, int(max_chars))
    if len(limited) > max_chars:
        limited = limited[:max_chars].rstrip(" ，。！？；：、,.!?;:;-—") + "。"
    return limited


def reset_indextts_cache():
    return None


def _import_edge_tts():
    import edge_tts

    return edge_tts


def _import_requests():
    import requests

    return requests


def _run_async(coro):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _edge_tts_output_path(output_path):
    path = Path(output_path)
    if path.suffix.lower() == ".mp3":
        return str(path)
    return str(path.with_suffix(".mp3"))


def _run_edge_tts(text, output_path, voice_config, fallback=False):
    output_path = _edge_tts_output_path(output_path)
    voice = voice_config.get("edge_voice", voice_config.get("voice", "zh-CN-XiaoxiaoNeural"))
    rate = voice_config.get("edge_rate", voice_config.get("rate", "+8%"))
    pitch = voice_config.get("edge_pitch", voice_config.get("pitch", "+8Hz"))
    volume = voice_config.get("edge_volume", voice_config.get("volume", "+0%"))

    async def generate():
        edge_tts = _import_edge_tts()
        communicate = edge_tts.Communicate(
            text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume,
        )
        await communicate.save(output_path)

    _run_async(generate())
    return {
        "engine": "edge_tts",
        "fallback": fallback,
        "output_path": output_path,
        "voice": voice,
    }


def _build_indextts_payload(text, output_path, voice_config):
    model_dir = resolve_project_path(voice_config.get("indextts_model_dir", get_default_indextts_model_dir()))
    cfg_path = resolve_project_path(voice_config.get("indextts_cfg_path", get_default_indextts_cfg_path()))
    speaker_audio = voice_config.get("indextts_speaker_audio")
    if not speaker_audio:
        raise ValueError("Index-TTS needs voice_settings.indextts_speaker_audio.")
    speaker_audio = resolve_project_path(speaker_audio)
    return {
        "text": text,
        "output_path": str(Path(output_path).with_suffix(".wav")),
        "model_dir": model_dir,
        "cfg_path": cfg_path,
        "speaker_audio": speaker_audio,
        "use_fp16": bool(voice_config.get("indextts_use_fp16", False)),
        "use_cuda_kernel": bool(voice_config.get("indextts_use_cuda_kernel", False)),
        "use_deepspeed": bool(voice_config.get("indextts_use_deepspeed", False)),
    }


def _post_indextts_service(service_url, payload, timeout):
    requests = _import_requests()

    response = requests.post(service_url, json=payload, timeout=timeout)
    if not response.ok:
        raise RuntimeError(f"{response.status_code} {response.reason}: {response.text}")
    return response.json()


def _run_indextts(text, output_path, voice_config):
    service_url = voice_config.get("indextts_service_url", "http://127.0.0.1:7862/tts")
    timeout = int(voice_config.get("indextts_timeout_seconds", 180))
    payload = _build_indextts_payload(text, output_path, voice_config)
    data = _post_indextts_service(service_url, payload, timeout)
    return {
        "engine": "indextts",
        "fallback": False,
        "output_path": data.get("output_path", payload["output_path"]),
        "service_url": service_url,
    }


def generate_tts_audio(text, output_path, voice_config):
    voice_config = voice_config or {}
    engine = str(voice_config.get("engine", "edge_tts")).lower()

    if engine == "indextts":
        try:
            return _run_indextts(text, output_path, voice_config)
        except Exception as exc:
            if str(voice_config.get("fallback_engine", "edge_tts")).lower() != "edge_tts":
                raise
            print(f"Index-TTS 生成失败，回退 edge-tts: {exc}")
            fallback_output_path = _edge_tts_output_path(output_path)
            return _run_edge_tts(text, fallback_output_path, voice_config, fallback=True)

    return _run_edge_tts(text, output_path, voice_config, fallback=False)
