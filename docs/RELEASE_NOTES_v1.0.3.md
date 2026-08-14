# Xinbao v1.0.3 Release Notes

这是 `v1.0.2` 的安装器热修复版，重点修复真实 release 测试中发现的 uv 托管 Python 环境缺少 pip 的问题。

## 修复内容

- 修复 `v1.0.2` 在无系统 Python 3.10/3.11 的电脑上，使用 uv 创建 `.venv` 后缺少 pip 的问题。
- `tools/bootstrap_release.ps1` 的 `uv venv` 命令现在使用 `--seed`，确保新环境自带 pip/setuptools/wheel 引导能力。
- README 和 `install.bat` 版本号更新到 `v1.0.3`。
- 默认语音路线仍是 `edge-tts`，Index-TTS 继续作为可选高音质路线保留，不进入默认安装链路。

## 用户启动方式

普通用户仍然按 README 的三步走：

```bat
install.bat
download_models.bat
start_xinbao_desktop.bat
```

首次运行 `install.bat` 仍需要联网下载 uv、本地 Python、项目依赖和 PyTorch wheel。网络不稳定时可以重新运行脚本，安装器会复用已经下载好的 `.python/`、`.tools/`、`.uv-cache/` 和 `.venv/` 内容。

如果本地已经存在旧的、不完整的 `.venv`，并且它不是 Python 3.10/3.11，安装器会提示用户清理后重试；这是为了避免自动删除用户目录里的内容。正常从 GitHub 新下载的 release 目录不会遇到这个问题。
