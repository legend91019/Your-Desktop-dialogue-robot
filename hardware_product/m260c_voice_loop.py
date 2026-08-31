#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M260C -> record -> ASR/manual text -> Xinbao chat loop.

The first goal is to run the whole robot interaction path safely:
wake word -> recording -> text input -> /api/chat.

No third-party Python packages are required. Optional Baidu ASR can be enabled
with environment variables:
  BAIDU_ASR_APP_ID
  BAIDU_ASR_API_KEY
  BAIDU_ASR_SECRET_KEY
"""

import argparse
import base64
import errno
import glob
import json
import os
import re
import select
import shutil
import struct
import subprocess
import sys
import termios
import time
import urllib.parse
import urllib.request
from pathlib import Path


SYNC_HEADER = 0xA5
USER_ID = 0x01
HANDSHAKE_MSG_TYPE = 0x01
DEVICE_MSG_TYPE = 0x04
CONTROL_MSG_TYPE = 0x05
HANDSHAKE_ACK_TYPE = 0xFF
HEADER_SIZE = 7
LOCAL_ENV_FILES = (
    os.path.join(os.path.dirname(__file__), "baidu_asr.env"),
    os.path.join(os.getcwd(), "baidu_asr.env"),
)


def checksum(data):
    return ((~(sum(data) & 0xFF)) + 1) & 0xFF


def encode_message(msg_type, payload, msg_id):
    header = bytearray([SYNC_HEADER, USER_ID, msg_type])
    header += struct.pack("<H", len(payload))
    header += struct.pack("<H", msg_id)
    frame = header + payload
    frame.append(checksum(frame))
    return bytes(frame)


def safe_write(fd, data):
    while data:
        try:
            written = os.write(fd, data)
            data = data[written:]
        except BlockingIOError as exc:
            if exc.errno != errno.EAGAIN:
                raise
            time.sleep(0.05)


def send_json(fd, msg_id, obj):
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    safe_write(fd, encode_message(CONTROL_MSG_TYPE, payload, msg_id))
    return msg_id + 1


def set_raw_115200(fd):
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = termios.B115200
    attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 5
    termios.tcsetattr(fd, termios.TCSANOW, attrs)


def list_ports():
    ports = []
    for pattern in ("/dev/ttyCH343USB*", "/dev/ttyUSB*", "/dev/ttyACM*"):
        ports.extend(glob.glob(pattern))
    return sorted(set(ports))


def open_first_working_port(ports):
    last_error = None
    for port in ports:
        try:
            fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
            set_raw_115200(fd)
            print(f"[M260C] Serial opened: {port}")
            return port, fd
        except Exception as exc:
            last_error = exc
            print(f"[M260C] Cannot open {port}: {exc}")
    raise RuntimeError(f"No usable serial port. Last error: {last_error}")


def wait_for_serial_port(args):
    while True:
        ports = [args.port] if args.port else list_ports()
        if ports:
            print("[M260C] Candidate ports:", " ".join(ports))
            try:
                return open_first_working_port(ports)
            except Exception as exc:
                print(f"[M260C] Serial open failed: {exc}")
        else:
            print("[M260C] No serial port found yet.")
            print("[M260C] Try: ls /dev/ttyCH343USB* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null")

        print("[M260C] Waiting 5s and retrying...")
        time.sleep(5)


def parse_frame(frame):
    if len(frame) < HEADER_SIZE + 1:
        return None
    if frame[0] != SYNC_HEADER or frame[1] != USER_ID:
        return None
    payload_len = struct.unpack("<H", frame[3:5])[0]
    total_len = HEADER_SIZE + payload_len + 1
    if len(frame) != total_len:
        return None
    if checksum(frame[:-1]) != frame[-1]:
        return None
    return {
        "type": frame[2],
        "length": payload_len,
        "id": struct.unpack("<H", frame[5:7])[0],
        "payload": frame[HEADER_SIZE:-1],
    }


def compact_wakeup(obj):
    content = obj.get("content", {})
    info = content.get("info")
    info_obj = {}
    if isinstance(info, str):
        try:
            info_obj = json.loads(info)
        except Exception:
            info_obj = {}
    ivw = info_obj.get("ivw", {}) if isinstance(info_obj, dict) else {}
    return {
        "result": content.get("result"),
        "keyword": ivw.get("keyword") or content.get("result"),
        "score": ivw.get("score"),
        "angle": ivw.get("angle"),
        "beam": ivw.get("beam") or content.get("arg1"),
    }


def is_wakeup(obj):
    return obj.get("type") == "aiui_event" and obj.get("content", {}).get("eventType") == 4


def post_json(url, payload, timeout=10):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def post_json_with_retry(url, payload, timeout=10, retries=3, delay=1.5):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return post_json(url, payload, timeout=timeout)
        except Exception as exc:
            last_exc = exc
            print(f"[Xinbao] HTTP attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(delay * attempt)
    raise last_exc


def notify_wakeup(api_base, event):
    try:
        body = post_json_with_retry(f"{api_base}/api/wakeup", event, timeout=3, retries=2)
        print(f"[Xinbao] Wakeup posted: {body[:160]}")
    except Exception as exc:
        print(f"[Xinbao] Wakeup post failed: {exc}")


def cleanup_old_files(directory, pattern, keep=30):
    try:
        files = sorted(
            Path(directory).glob(pattern),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
    except Exception as exc:
        print(f"[Cleanup] Cannot scan {directory}/{pattern}: {exc}")
        return

    for old_file in files[keep:]:
        try:
            old_file.unlink()
            print(f"[Cleanup] Removed old file: {old_file}")
        except Exception as exc:
            print(f"[Cleanup] Cannot remove {old_file}: {exc}")


def record_wav(path, seconds, device=None):
    if not shutil.which("arecord"):
        raise RuntimeError("arecord not found. Install alsa-utils or use the desktop recording path.")

    cmd = ["arecord"]
    if device:
        cmd += ["-D", device]
    cmd += ["-f", "S16_LE", "-r", "16000", "-c", "1", "-d", str(seconds), path]

    print("[Audio] Recording:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    size = os.path.getsize(path)
    print(f"[Audio] Saved: {path} ({size} bytes)")
    return path


def load_local_env_files():
    for env_path in LOCAL_ENV_FILES:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw_line in f:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
            print(f"[ASR] Loaded local env file: {env_path}")
        except Exception as exc:
            print(f"[ASR] Failed to load env file {env_path}: {exc}")


def baidu_token(api_key, secret_key):
    query = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "client_id": api_key,
        "client_secret": secret_key,
    })
    url = f"https://aip.baidubce.com/oauth/2.0/token?{query}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if "access_token" not in data:
        raise RuntimeError(f"Baidu token error: {data}")
    return data["access_token"]


def baidu_asr(wav_path):
    app_id = os.environ.get("BAIDU_ASR_APP_ID", "").strip()
    api_key = os.environ.get("BAIDU_ASR_API_KEY", "").strip()
    secret_key = os.environ.get("BAIDU_ASR_SECRET_KEY", "").strip()
    if not app_id or not api_key or not secret_key:
        return None

    token = baidu_token(api_key, secret_key)
    with open(wav_path, "rb") as f:
        speech = base64.b64encode(f.read()).decode("ascii")
    payload = {
        "format": "wav",
        "rate": 16000,
        "channel": 1,
        "cuid": "xinbao-orange-pi",
        "token": token,
        "dev_pid": 1537,
        "speech": speech,
        "len": os.path.getsize(wav_path),
    }
    req = urllib.request.Request(
        "https://vop.baidu.com/server_api",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    if result.get("err_no") != 0:
        raise RuntimeError(f"Baidu ASR error: {result}")
    texts = result.get("result") or []
    return texts[0].strip("，。,. ") if texts else ""


def manual_transcript(wav_path):
    print("\n[ASR] No Baidu ASR env configured, entering manual transcript mode.")
    print(f"[ASR] Recording file: {wav_path}")
    print("[ASR] Type the sentence you just spoke, then press Enter.")
    try:
        return input("你刚才说的是: ").strip()
    except EOFError:
        return ""


def extract_chat_text_from_sse(body):
    chunks = []
    audio_url = None
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        try:
            obj = json.loads(line[6:])
        except Exception:
            continue
        if obj.get("chunk"):
            chunks.append(obj["chunk"])
        if obj.get("audio_url"):
            audio_url = obj["audio_url"]
    text = "".join(chunks)
    text = re.sub(r"\[.*?\]", "", text).strip()
    return text, audio_url


def pebble_v3_available():
    try:
        result = subprocess.run(["aplay", "-l"], text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except Exception:
        return False
    return "V3 [Pebble V3]" in result.stdout or "Pebble V3" in result.stdout


def play_tts_audio(audio_url):
    if not audio_url:
        return
    parsed = urllib.parse.urlparse(audio_url)
    rel_path = parsed.path if parsed.scheme else audio_url
    rel_path = rel_path.lstrip('/')
    audio_path = os.path.join(os.path.dirname(__file__), rel_path)
    if not os.path.exists(audio_path):
        print(f"[AudioOut] TTS file not found: {audio_path}")
        return

    if pebble_v3_available() and shutil.which("ffmpeg") and shutil.which("aplay"):
        print("[AudioOut] Playing on Pebble V3: ffmpeg | aplay -D plughw:CARD=V3,DEV=0")
        ffmpeg_cmd = ["ffmpeg", "-v", "error", "-i", audio_path, "-f", "wav", "-"]
        aplay_cmd = ["aplay", "-q", "-D", "plughw:CARD=V3,DEV=0"]
        try:
            ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE)
            aplay_proc = subprocess.run(aplay_cmd, stdin=ffmpeg_proc.stdout, check=False)
            if ffmpeg_proc.stdout:
                ffmpeg_proc.stdout.close()
            ffmpeg_proc.wait()
            if aplay_proc.returncode == 0:
                return
            print(f"[AudioOut] Pebble playback exited with code {aplay_proc.returncode}; falling back to ffplay.")
        except Exception as exc:
            print(f"[AudioOut] Pebble playback failed: {exc}; falling back to ffplay.")

    if not shutil.which("ffplay"):
        print("[AudioOut] ffplay not found, skip playback.")
        return
    cmd = ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", audio_path]
    print("[AudioOut] Playing:", " ".join(cmd))
    try:
        subprocess.run(cmd, check=False)
    except Exception as exc:
        print(f"[AudioOut] Playback failed: {exc}")


def send_to_chat(api_base, text):
    print(f"[Xinbao] Sending text to chat: {text}")
    body = post_json_with_retry(f"{api_base}/api/chat", {"message": text}, timeout=90, retries=3)
    reply, audio_url = extract_chat_text_from_sse(body)
    print(f"[Xinbao] Reply: {reply or '(stream parsed, but no text chunk found)'}")
    if audio_url:
        print(f"[Xinbao] TTS audio: {api_base}{audio_url}")
        play_tts_audio(audio_url)
    return reply, audio_url


def drain_serial_input(fd, seconds=1.5):
    end = time.time() + seconds
    total = 0
    while time.time() < end:
        ready, _, _ = select.select([fd], [], [], 0.05)
        if not ready:
            continue
        try:
            total += len(os.read(fd, 4096))
        except BlockingIOError:
            pass
    print(f"[M260C] Drained serial input: {total} bytes")


def init_m260c_module(fd):
    drain_serial_input(fd)
    init_messages = [
        {"type": "switch_mic", "content": {"mic": "mic6_circle"}},
        {"type": "version"},
        {"type": "wakeup_keywords", "content": {"keyword": "xin1 bao3 xin1 bao3", "threshold": "500"}},
    ]
    msg_id = 1
    for msg in init_messages:
        msg_id = send_json(fd, msg_id, msg)
        print(f"[M260C] Init sent: {msg.get('type')}")
        time.sleep(0.5)
    return msg_id

def wait_for_wakeup(fd, api_base, manual=False):
    if manual:
        input("[Manual] Press Enter to simulate wakeup and start recording...")
        event = {
            "result": "manual",
            "keyword": "manual",
            "score": None,
            "angle": None,
            "beam": None,
        }
        notify_wakeup(api_base, event)
        return event

    buffer = b""
    last_hint = time.time()
    while True:
        ready, _, _ = select.select([fd], [], [], 0.5)
        if ready:
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                chunk = b""
            if chunk:
                buffer += chunk

        while len(buffer) >= HEADER_SIZE + 1:
            sync_pos = buffer.find(bytes([SYNC_HEADER]))
            if sync_pos < 0:
                buffer = b""
                break
            if sync_pos > 0:
                buffer = buffer[sync_pos:]
            if len(buffer) < HEADER_SIZE:
                break

            payload_len = struct.unpack("<H", buffer[3:5])[0]
            total_len = HEADER_SIZE + payload_len + 1
            if len(buffer) < total_len:
                break

            raw = buffer[:total_len]
            buffer = buffer[total_len:]
            frame = parse_frame(raw)
            if not frame:
                continue

            if frame["type"] == HANDSHAKE_MSG_TYPE:
                ack_payload = bytes([0xA5, 0x00, 0x00, 0x00])
                safe_write(fd, encode_message(HANDSHAKE_ACK_TYPE, ack_payload, frame["id"]))
                continue

            if frame["type"] != DEVICE_MSG_TYPE:
                continue

            text = frame["payload"].decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except Exception:
                continue

            if obj.get("type") == "version":
                print(f"[M260C] Version: {obj.get('content')}")
            elif is_wakeup(obj):
                event = compact_wakeup(obj)
                print(f"\n[M260C] Wakeup detected: {event}")
                notify_wakeup(api_base, event)
                return event
            else:
                print(f"[M260C] Device message: {text[:300]}")

        if time.time() - last_hint > 20:
            print("[M260C] Waiting for wakeup. Say: 芯宝芯宝")
            last_hint = time.time()


def main():
    load_local_env_files()

    parser = argparse.ArgumentParser(description="Xinbao M260C voice loop")
    parser.add_argument("-p", "--port", help="M260C serial port, e.g. /dev/ttyACM0")
    parser.add_argument("--api-base", default="http://127.0.0.1:5000")
    parser.add_argument("--record-seconds", type=int, default=5)
    parser.add_argument("--audio-device", default="default", help="arecord device, e.g. default or plughw:1,0")
    parser.add_argument("--record-dir", default="voice_records")
    parser.add_argument("--keep-audio-files", type=int, default=30, help="number of newest wav/mp3 files to keep")
    parser.add_argument("--once", action="store_true", help="handle one wakeup and exit")
    parser.add_argument("--manual", action="store_true", help="press Enter to trigger recording instead of waiting for M260C wakeup")
    parser.add_argument("--skip-serial", action="store_true", help="do not open M260C serial; implies --manual")
    args = parser.parse_args()

    os.makedirs(args.record_dir, exist_ok=True)

    fd = None
    if args.skip_serial:
        args.manual = True
        print("[Manual] Serial skipped. Press Enter to start each recording.")
    else:
        port, fd = wait_for_serial_port(args)
        print("[M260C] Voice loop ready. Say: 芯宝芯宝")

        time.sleep(0.2)
        init_m260c_module(fd)

    try:
        while True:
            wait_for_wakeup(fd, args.api_base, manual=args.manual)

            print(f"[Audio] Start speaking after the beep. Recording {args.record_seconds}s.")
            sys.stdout.write("\a")
            sys.stdout.flush()
            time.sleep(0.6)

            ts = time.strftime("%Y%m%d_%H%M%S")
            wav_path = os.path.join(args.record_dir, f"xinbao_{ts}.wav")
            try:
                record_wav(wav_path, args.record_seconds, args.audio_device)
            except Exception as exc:
                print(f"[Audio] Recording failed: {exc}")
                if args.once:
                    return 1
                continue

            try:
                transcript = baidu_asr(wav_path)
            except Exception as exc:
                print(f"[ASR] Baidu ASR failed: {exc}")
                transcript = None

            if transcript is None:
                transcript = manual_transcript(wav_path)

            if not transcript:
                print("[ASR] Empty transcript, skip chat.")
                if args.once:
                    return 1
                continue

            try:
                send_to_chat(args.api_base, transcript)
            except Exception as exc:
                print(f"[Xinbao] Chat request failed: {exc}")

            cleanup_old_files(args.record_dir, "*.wav", keep=args.keep_audio_files)
            cleanup_old_files("static", "reply_*.mp3", keep=args.keep_audio_files)

            if args.once:
                return 0

            print("\n[M260C] Back to standby. Say: 芯宝芯宝")
    except KeyboardInterrupt:
        print("\n[VoiceLoop] Stopped.")
    finally:
        if fd is not None:
            os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
