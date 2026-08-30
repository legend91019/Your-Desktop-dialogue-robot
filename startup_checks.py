"""Preflight checks used by the desktop launcher."""

import hashlib
import json
import socket
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckResult:
    ok: bool
    code: str
    message: str
    details: dict = field(default_factory=dict)


def check_gpu(min_vram_mb: int = 4096) -> CheckResult:
    try:
        import torch
    except Exception as exc:
        return CheckResult(False, "torch_unavailable", "PyTorch 加载失败，请重新安装芯宝。", {"error": str(exc)})

    if not torch.cuda.is_available():
        return CheckResult(False, "cuda_unavailable", "未检测到可用的 NVIDIA CUDA 环境。", {})

    try:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        vram_mb = int(props.total_memory / (1024 * 1024))
        details = {"device": props.name, "vram_mb": vram_mb, "cuda": torch.version.cuda}
        if vram_mb < min_vram_mb:
            return CheckResult(False, "vram_low", f"显存不足，建议至少 {min_vram_mb // 1024} GB。", details)
        return CheckResult(True, "gpu_ok", "NVIDIA GPU 检查通过。", details)
    except Exception as exc:
        return CheckResult(False, "gpu_check_failed", "读取 NVIDIA GPU 信息失败。", {"error": str(exc)})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_models(root: Path) -> CheckResult:
    root = Path(root)
    required = {
        "embedding": root / "models" / "embedding",
        "reranker": root / "models" / "reranker",
        "classifier": root / "assets" / "classifier" / "route_classifier.joblib",
        "knowledge": root / "knowledge.md",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        return CheckResult(False, "models_missing", f"模型或知识库缺失：{', '.join(missing)}。", {"missing": missing})

    manifest_path = root / "models" / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            failures = []
            for item in manifest.get("files", []):
                path = root / item["path"]
                if not path.exists() or _sha256(path) != item.get("sha256"):
                    failures.append(item["path"])
            if failures:
                return CheckResult(False, "models_checksum_failed", "模型校验失败，请重新安装芯宝。", {"files": failures})
        except (OSError, ValueError, KeyError) as exc:
            return CheckResult(False, "manifest_invalid", "模型清单无法读取，请重新安装芯宝。", {"error": str(exc)})

    return CheckResult(True, "models_ok", "本地模型和知识库检查通过。", {})


def find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def run_all(project_root: Path) -> list[CheckResult]:
    return [check_gpu(), check_models(project_root)]
