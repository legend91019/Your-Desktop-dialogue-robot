# Xinbao v1.0.4 Release Notes

这是面向默认 `edge-tts` release 路线的安装体验修复版。

## 修复内容

- 默认 PyTorch 安装从 CUDA 12.8 改为 CPU wheel，避免普通用户首次运行 `install.bat` 时下载约 2.7GB 的 CUDA 版 torch。
- CUDA 12.8 仍然保留为可选项，需要时可以在 PowerShell 里先设置：

```powershell
$env:XINBAO_TORCH_VARIANT="cu128"
```

然后再运行：

```bat
install.bat
```

- README 和 `install.bat` 版本号更新到 `v1.0.4`。
- `Index-TTS` 仍然是可选高音质路线，不进入默认依赖安装和模型下载链路。

## 用户启动方式

普通用户仍然按三步走：

```bat
install.bat
download_models.bat
start_xinbao_desktop.bat
```

这版仍会在没有 Python 3.10/3.11 的电脑上自动下载本地 Python 3.11，并创建项目私有 `.venv/`。本地运行产物包括 `.python/`、`.tools/`、`.uv-cache/`、`.venv/`，都已加入 `.gitignore`，不要上传到公开仓库。
