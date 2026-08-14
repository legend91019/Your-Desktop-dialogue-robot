# Xinbao v1.0.2 Release Notes

这是 `v1.0.1` 的 release 安装器修复版，目标是让普通 Windows 用户尽量只按 README 的三步脚本完成启动：

```bat
install.bat
download_models.bat
start_xinbao_desktop.bat
```

## 修复内容

- `install.bat` 改为调用 `tools/bootstrap_release.ps1`，由 PowerShell 统一处理 release 环境准备。
- 安装器会优先复用系统已有的 Python 3.11 / 3.10。
- 如果用户电脑只有 Python 3.13，或没有可用的兼容 Python，安装器会通过 uv 自动创建项目本地 Python 3.11 运行时。
- 本地 Python 会放在 `.python/`，uv 工具会放在 `.tools/`，uv 缓存会放在 `.uv-cache/`。
- `.python/`、`.tools/`、`.uv-cache/` 已加入 `.gitignore`，避免把 release 运行产物上传到公开仓库。
- 默认语音路线仍是 `edge-tts`，Index-TTS 继续保持可选，不进入默认下载和安装链路。

## 注意

首次运行 `install.bat` 仍需要联网下载依赖、PyTorch wheel，以及在缺少兼容 Python 时下载 uv/托管 Python。网络不稳定时可以重新运行 `install.bat`，脚本会复用已经下载好的本地环境。
