# 芯宝 Xinbao：桌面端对话机器人

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Backend](https://img.shields.io/badge/Backend-Flask-green)
![Memory](https://img.shields.io/badge/Memory-ChromaDB-orange)
![Release](https://img.shields.io/badge/Release-v1.0.0-brightgreen)

芯宝是一个面向个人使用的桌面对话机器人项目，包含云端大模型对话、本地向量记忆、长期记忆管理、edge-tts 语音播报，以及可选的 Index-TTS 本地音色方案。

## 快速开始

第一次运行推荐按顺序执行：

```powershell
install.bat
download_models.bat
start_xinbao_desktop.bat
```

如果你使用 conda，也可以先激活自己的环境：

```powershell
conda activate xinbao_env
```

## v1.0.0 的语音策略

本版本提供两条路线：

| 路线 | 默认状态 | 适合人群 | 额外要求 |
| --- | --- | --- | --- |
| edge-tts | 默认启用 | 想快速运行和演示的用户 | 基础 Python 环境 |
| Index-TTS | 可选启用 | 想使用参考音频和更自然音色的用户 | 官方仓库、独立环境、模型和参考音频 |

不配置 Index-TTS 时，芯宝仍然可以使用 edge-tts 正常运行。Index-TTS 不属于基础安装，也不会被默认模型下载脚本下载。

## 基础路线：edge-tts

### 1. 安装依赖

```powershell
install.bat
```

等价于：

```powershell
python -m pip install -r requirements.txt
```

### 2. 下载基础模型

```powershell
download_models.bat
```

等价于：

```powershell
python download_all_models.py
```

默认只下载 embedding 和 reranker，不下载 Index-TTS。

如果还需要训练本地分类器，可以运行：

```powershell
models_ready.bat
```

### 3. 配置 API Key

请参考 `config.example.json` 创建或修改本地 `config.json`，填写自己的 DeepSeek API Key。

公开仓库中的 `config.json` 和 `config.example.json` 默认都不包含真实 API Key，并且默认使用：

```json
{
  "engine": "edge_tts",
  "fallback_engine": "edge_tts"
}
```

真实 API Key、数据库、模型和运行缓存不能提交到 GitHub。

### 4. 启动桌面版

```powershell
start_xinbao_desktop.bat
```

或者：

```powershell
python desktop_launcher.py
```

基础路线不需要下载 Index-TTS 官方仓库，也不需要创建 Index-TTS 专用环境。

## 可选增强：Index-TTS 本地音色路线

如果希望使用参考音频生成更有角色感的声音，可以额外配置 Index-TTS。它不是本项目的强制依赖。

### 1. 下载官方仓库

请按照 Index-TTS 官方文档下载源码：

<https://github.com/index-tts/index-tts>

建议把官方仓库放在不含中文的路径，例如：

```text
D:\IndexTTS\index-tts-src
```

不要把 Index-TTS 源码复制到本项目仓库中。

### 2. 创建独立环境

在 Index-TTS 官方仓库目录中，按照官方文档创建它自己的 `.venv`。

不要把 Index-TTS 的依赖混装进芯宝主环境。Windows 用户第一次配置时，建议先走官方基础安装流程，不要默认安装可能需要本地编译的 `flash-attn` 加速依赖。

详细说明见：

[`docs/indextts_local_setup.md`](docs/indextts_local_setup.md)

### 3. 下载可选模型

在芯宝项目根目录执行：

```powershell
python download_all_models.py --with-indextts
```

Index-TTS 模型会下载到：

```text
models\IndexTTS\checkpoints
```

模型目录已经被 `.gitignore` 排除，不要提交模型权重。

### 4. 放置参考音频

准备一段 5～15 秒、单人、无背景音乐、较干净的女声 WAV，放到：

```text
models\IndexTTS\xinbao_voice.wav
```

### 5. 启动 Index-TTS 服务

使用 Index-TTS 官方环境启动芯宝提供的服务脚本：

```powershell
cd /d D:\IndexTTS\index-tts-src
.venv\Scripts\python.exe "D:\path\to\IntelliChat-Platform\tools\indextts_service.py" --host 127.0.0.1 --port 7862 --indextts-src "D:\IndexTTS\index-tts-src"
```

看到下面的提示，说明服务已经开始监听：

```text
Index-TTS service listening on http://127.0.0.1:7862
```

### 6. 切换配置

在本地 `config.json` 的 `voice_settings` 中设置：

```json
{
  "engine": "indextts",
  "fallback_engine": "edge_tts",
  "indextts_service_url": "http://127.0.0.1:7862/tts",
  "indextts_model_dir": "models\\IndexTTS\\checkpoints",
  "indextts_speaker_audio": "models\\IndexTTS\\xinbao_voice.wav"
}
```

然后重新启动桌面版。如果 Index-TTS 服务未启动、模型缺失或生成超时，后端会回退到 edge-tts。

## 记忆管理

芯宝支持：

- 查看长期记忆
- 手动添加记忆
- 修改记忆内容
- 删除错误或过期记忆

长期记忆保存在本地 ChromaDB 中。`chroma_db/` 属于运行数据，不应提交到公开仓库。

## 项目结构

```text
IntelliChat-Platform/
├─ BackEnd/
│  ├─ simple.py              # Flask 后端与对话流程
│  ├─ memory_admin.py        # 长期记忆管理
│  ├─ tts_engine.py          # edge-tts / Index-TTS 路由
│  └─ audio_player.py        # 桌面端音频播放辅助
├─ FrontEnd/robot.html       # 对话页面
├─ tools/indextts_service.py # 可选 Index-TTS 服务
├─ tests/                    # 软件端测试
├─ config.example.json       # 公共示例配置
├─ install.bat               # 安装基础依赖
├─ download_models.bat       # 下载基础模型
├─ download_all_models.py    # 模型下载脚本
├─ desktop_launcher.py       # 桌面启动器
├─ models_ready.bat          # 基础模型与分类器初始化
└─ start_xinbao_desktop.bat  # 桌面启动
```

## 发布安全规则

以下内容不能上传 GitHub：

- API Key、Token、密码
- `models/` 模型权重
- `chroma_db/` 本地数据库
- `static/` 中的运行音频
- `voice_records/` 录音
- 运行日志和临时缓存
- 个人参考音频
- 查新报告、研究报告等非代码材料
- Index-TTS 官方仓库及其独立环境

## 当前版本

`v1.0.0`

这是芯宝桌面软件端的第一个正式 GitHub Release。该版本优先保证默认路线可安装、可启动、仓库干净，并把 Index-TTS 保持为可选增强。
