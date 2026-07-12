#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bridge M260C wakeup events into Xinbao.

This script listens to the WHEELTEC M2 serial protocol. When it receives
aiui_event/eventType=4, it posts a compact wakeup event to Xinbao's Flask API.
No third-party Python packages are required.
"""

import argparse
import glob
import json
import os
import select
import struct
import sys
import termios
import time
import urllib.error
import urllib.request


SYNC_HEADER = 0xA5
USER_ID = 0x01
HANDSHAKE_MSG_TYPE = 0x01
DEVICE_MSG_TYPE = 0x04
CONTROL_MSG_TYPE = 0x05
HANDSHAKE_ACK_TYPE = 0xFF
HEADER_SIZE = 7


def checksum(data):
    return ((~(sum(data) & 0xFF)) + 1) & 0xFF


def encode_message(msg_type, payload, msg_id):
    header = bytearray([SYNC_HEADER, USER_ID, msg_type])
    header += struct.pack("<H", len(payload))
    header += struct.pack("<H", msg_id)
    frame = header + payload
    frame.append(checksum(frame))
    return bytes(frame)


def send_json(fd, msg_id, obj):
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    os.write(fd, encode_message(CONTROL_MSG_TYPE, payload, msg_id))
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


def post_wakeup(api_url, event):
    data = json.dumps(event, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        api_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
        print(f"[M260C] Posted wakeup to Xinbao: {body}")
        return True
    except urllib.error.URLError as exc:
        print(f"[M260C] Cannot post to Xinbao API: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="M260C wakeup bridge for Xinbao")
    parser.add_argument("-p", "--port", help="serial port, for example /dev/ttyACM0")
    parser.add_argument("--api", default="http://127.0.0.1:5000/api/wakeup")
    args = parser.parse_args()

    ports = [args.port] if args.port else list_ports()
    if not ports:
        print("[M260C] No serial port found.")
        print("[M260C] Try: ls /dev/ttyCH343USB* /dev/ttyUSB* /dev/ttyACM* 2>/dev/null")
        return 2

    print("[M260C] Candidate ports:", " ".join(ports))
    port, fd = open_first_working_port(ports)
    print("[M260C] Listening. Say: 小微小微")

    msg_id = 1
    time.sleep(0.2)
    msg_id = send_json(fd, msg_id, {"type": "version"})

    buffer = b""
    try:
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
                    os.write(fd, encode_message(HANDSHAKE_ACK_TYPE, ack_payload, frame["id"]))
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
                    print(f"[M260C] Wakeup detected on {port}: {event}")
                    post_wakeup(args.api, event)
    except KeyboardInterrupt:
        print("\n[M260C] Bridge stopped.")
    finally:
        os.close(fd)


if __name__ == "__main__":
    raise SystemExit(main())
