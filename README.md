<p align="center">
  <img src="assets/branding/xinbao-logo.jpg" alt="芯宝 Xinbao logo" width="260">
</p>

<h1 align="center">芯宝 Xinbao</h1>

<p align="center">面向 Windows 的本地记忆对话陪伴机器人</p>

<p align="center">
  <a href="https://github.com/legend91019/Your-Desktop-dialogue-robot/releases"><img src="https://img.shields.io/github/v/release/legend91019/Your-Desktop-dialogue-robot?display_name=tag&sort=semver" alt="Latest release"></a>
  <a href="https://github.com/legend91019/Your-Desktop-dialogue-robot/releases"><img src="https://img.shields.io/github/downloads/legend91019/Your-Desktop-dialogue-robot/total" alt="Downloads"></a>
  <img src="https://img.shields.io/badge/Windows-10%20%2F%2011-0078D4" alt="Windows 10 / 11">
  <img src="https://img.shields.io/badge/NVIDIA-CUDA%2012.x-76B900" alt="NVIDIA CUDA 12.x">
</p>

芯宝把云端对话、本地长期记忆和 RAG 检索放进一个桌面窗口。普通用户使用发布版安装器即可启动，不需要安装 Conda、Python、pip 或手工下载模型；只需在设置面板填写自己的 DeepSeek API Key。

## 能做什么

- DeepSeek 流式对话
- ChromaDB 长期记忆与本地知识检索
- 可管理的记忆记录和用户上传内容
- Windows 桌面窗口与动态本地端口
- 默认 `edge-tts` 语音输出，支持可选 Index-TTS 服务
- 启动前 GPU、模型文件和校验值检查

## 下载桌面版

代码、发布说明和 SHA-256 校验文件位于 [GitHub Releases](https://github.com/legend91019/Your-Desktop-dialogue-robot/releases)。

完整 Windows 安装器包含 CUDA PyTorch 和本地模型，约 3.55 GiB，超过 GitHub 单个 Release 资产的 2 GiB 限制。安装器由项目维护者通过外部对象存储分发，下载地址会在对应 Release 说明中提供；不要从不明来源下载同名文件。

下载后可用 PowerShell 校验：

```powershell
Get-FileHash .\Xinbao-Setup-v1.0.7.exe -Algorithm SHA256
```

将结果与 Release 中的 `.sha256` 文件比较。安装器文件名中的版本号会随发布版本变化。

## 普通用户安装

1. Windows 10/11 64 位电脑安装支持 CUDA 12.x 的 NVIDIA 驱动。
2. 运行 `Xinbao-Setup-v<version>.exe`，按向导完成安装。
3. 从桌面快捷方式启动“芯宝 Xinbao”。
4. 首次打开，在设置面板填写 DeepSeek API Key 和主人信息并保存。

芯宝会把配置、API Key、长期记忆、上传文件、语音缓存和日志保存到：

```text
%APPDATA%\Xinbao\
```

升级安装不会覆盖这些用户数据；卸载默认只删除程序文件。

## 系统要求与网络边界

- Windows 10 或 Windows 11，64 位
- NVIDIA GPU，建议显存不低于 4 GB
- 支持 CUDA 12.x 的 NVIDIA 驱动
- DeepSeek 对话需要联网并使用用户自己的 API Key
- `edge-tts` 语音需要联网；语音失败不影响文字对话

首版不支持无 NVIDIA GPU 的普通安装路径。启动器会在窗口打开前报告 GPU、驱动、显存、模型缺失、端口和后端初始化问题。

## 开发者运行

开发者可以在已安装 CPython 3.10/3.11 的环境中使用项目脚本：

```bat
install.bat
download_models.bat
start_xinbao_desktop.bat
```

需要 Index-TTS 开发资源时，可运行 `python tools/download_all_models.py --with-indextts`；普通用户安装器已提供默认的 edge-tts 语音路径。

这些脚本只服务于开发和调试，不是普通用户的安装步骤。构建发布运行时使用项目私有 `.venv`，不读取 Conda 环境；`hardware_product` 保留为硬件版本目录，不属于当前桌面版发布范围。

运行测试：

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## 项目结构

```text
BackEnd/                 Flask/Waitress 后端、记忆和 TTS
FrontEnd/                桌面窗口加载的前端页面
utils/                   路由分类器与 RAG 检索
assets/branding/         芯宝品牌 logo
assets/classifier/       发布版轻量路由分类器
packaging/               Inno Setup 安装器与 Windows 图标
tools/                   构建、模型清单和开发辅助脚本
docs/                    用户指南、发布说明和验收清单
hardware_product/        硬件版本，当前不参与桌面发布
```

## 安全与隐私

请不要提交以下内容：

- 带真实 API Key 的 `config.json`
- `.venv/`、`.python/` 和 `build/`、`dist/` 构建产物
- `models/` 模型权重、`chroma_db/` 长期记忆和运行音频
- 个人报告、临时缓存和硬件测试数据

API Key 只保存在本机 `%APPDATA%\Xinbao\config.json`，日志不会记录 Key 明文。

发布包内置轻量路由分类器文件 `assets/classifier/route_classifier.joblib`，无需首次启动时训练。

## 发布版本

- `v1.0.7`：Windows 桌面安装版、私有 CPython 3.11 运行时、CUDA PyTorch、本地模型清单和品牌安装器
- `v1.0.6`：上一版桌面运行时基线，保留用于升级兼容性参考
- 详细变更见 [`docs/RELEASE_NOTES_v1.0.7.md`](docs/RELEASE_NOTES_v1.0.7.md)
- 用户操作见 [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)
