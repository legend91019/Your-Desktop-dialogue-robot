# Xinbao v1.0.1 Release Notes

这是 `v1.0.0` 的安装器修复版，主要解决首次 release 测试时暴露的依赖安装问题。

## 修复内容

- `install.bat` 现在会自动创建项目私有 `.venv`，不再默认把依赖安装到系统 Python。
- 安装器优先寻找 Python 3.11 / 3.10；如果只有 Python 3.13，会明确提示安装兼容版本。
- PyTorch 从 `requirements.txt` 中拆出，由安装器单独从 PyTorch 官方 wheel 源安装。
- `download_models.bat` 会使用 `.venv` 下载基础模型，并训练本地分类器。
- `download_all_models.py` 默认不下载 Index-TTS；只有传入 `--with-indextts` 时才下载可选 TTS 模型。
- README 改成正式 release 三步流程：`install.bat` → `download_models.bat` → `start_xinbao_desktop.bat`。

## 默认路线

默认语音仍为 `edge-tts`，以保证安装成功率和桌面演示稳定性。

Index-TTS 继续作为可选高音质路线保留，不进入默认依赖安装流程。

## 安全提醒

公开仓库不应包含：

- 真实 API Key；
- `models/` 模型权重；
- `chroma_db/` 本地记忆数据库；
- `static/reply_*.mp3` 运行音频；
- `.venv/` 虚拟环境；
- 个人报告和临时缓存。
