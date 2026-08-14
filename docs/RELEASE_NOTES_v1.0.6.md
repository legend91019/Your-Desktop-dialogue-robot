# Xinbao v1.0.6 Release Notes

这是面向 GPU 桌面演示路线的 release 策略修正。

## 调整内容

- 默认 PyTorch 安装策略从 CPU wheel 改回 CUDA 12.8 wheel：

```text
torch / torchvision / torchaudio --index-url https://download.pytorch.org/whl/cu128
```

- CPU 版 PyTorch 保留为 fallback，需要时可以在 PowerShell 里先设置：

```powershell
$env:XINBAO_TORCH_VARIANT="cpu"
```

然后再运行：

```bat
install.bat
```

- README 和 `install.bat` 版本号更新到 `v1.0.6`。
- 继续保留 `v1.0.5` 的内置轻量路由分类器，普通用户仍然不需要运行 `train_classifier.py`。

## 说明

芯宝当前的 RAG embedding、reranker，以及后续本地模型/高质量 TTS 路线都更适合 GPU 环境。因此默认 GPU 版 PyTorch 是更符合项目定位的选择。首次安装下载体积较大是 CUDA wheel 的正常代价。
