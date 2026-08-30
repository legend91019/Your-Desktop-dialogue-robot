# 芯宝 Xinbao：桌面对话陪伴机器人

芯宝是一个面向桌面演示与后续边缘设备部署的对话机器人项目。当前版本以 PC 桌面端为主，提供：

- DeepSeek 云端对话能力；
- 本地 ChromaDB 长程记忆与 RAG 检索；
- 可管理记忆的后端接口；
- 桌面窗口启动器；
- 默认 `edge-tts` 语音输出；
- 可选 Index-TTS 高音质语音方案。

本仓库的公开版本不应包含 API Key、本地数据库、运行音频、模型权重或个人报告材料。

当前版本：`v1.0.6`。这是面向 GPU 桌面演示的 release 版本：默认安装 CUDA 12.8 版 PyTorch，同时保留 v1.0.5 的内置轻量路由分类器，普通用户不需要训练分类器。

## 快速开始（普通用户）

正式发布版提供 `Xinbao-Setup-v<version>.exe` 安装程序。普通用户只需安装并双击桌面快捷方式，不需要安装 Python、Conda、pip，也不需要运行下面的开发者脚本。

安装包内置 CPython 3.11、项目私有运行环境、CUDA PyTorch、本地 RAG 模型和路由分类器。首版要求 Windows 10/11、NVIDIA GPU（建议显存 4 GB 以上）以及支持 CUDA 12.x 的驱动。DeepSeek 对话和 edge-tts 语音仍需要联网，API Key 由用户在设置面板自行填写。

详细说明见 [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md)。

## 开发者/构建者流程

开发者或未使用正式安装包的测试人员可以按这三个脚本运行：

```bat
install.bat
download_models.bat
start_xinbao_desktop.bat
```

### 1. 安装依赖

双击运行：

```bat
install.bat
```

安装脚本会自动在项目目录下创建私有环境：

```text
.venv/
```

后续脚本都会使用：

```text
.venv\Scripts\python.exe
```

这样不会把依赖装进用户的系统 Python、Anaconda base 环境或其它项目环境。

安装器会优先寻找电脑上已有的 Python 3.11 / 3.10。如果没有兼容版本，它会自动下载一个项目本地使用的 Python 3.11 运行时到：

```text
.python/
```

并使用它创建 `.venv/`。这两个目录都属于本地运行产物，不应该上传到 GitHub。

默认安装 CUDA 12.8 版 PyTorch，以匹配芯宝的本地 RAG、reranker 和后续 GPU 推理需求。这个 wheel 较大，首次安装下载几 GB 属于正常现象。

如果电脑没有 NVIDIA 显卡，只想用 CPU fallback，可以先在 PowerShell 里设置：

```powershell
$env:XINBAO_TORCH_VARIANT="cpu"
```

然后再运行 `install.bat`。

### 2. 下载基础模型

双击运行：

```bat
download_models.bat
```

该脚本会下载基础 RAG / 向量检索模型，并检查仓库内置的轻量路由分类器：

```text
assets/classifier/route_classifier.joblib
```

普通用户不需要运行 `train_classifier.py`。它不会下载 Index-TTS 大模型。

`models_ready.bat` 保留为旧流程兼容入口，作用与 `download_models.bat` 基本一致：

```bat
models_ready.bat
```

### 3. 启动桌面版

双击运行：

```bat
start_xinbao_desktop.bat
```

脚本会启动后端服务，并通过桌面窗口打开前端页面。

首次运行时，请在前端配置面板里填入 DeepSeek API Key 和主人信息。配置会保存到本地 `config.json`，不要把带真实 Key 的配置文件上传到公开仓库。

## 可选：Index-TTS 高音质语音

Release 默认使用 `edge-tts`，优先保证安装成功率和桌面演示稳定性。

如果需要启用 Index-TTS：

```bat
.venv\Scripts\python.exe download_all_models.py --with-indextts
```

然后参考：

```text
docs/indextts_local_setup.md
```

Index-TTS 建议使用它自己的官方仓库和独立环境，不要把 Index-TTS 的复杂依赖混进主项目 `.venv`。

## 项目结构

```text
BackEnd/                    后端服务、TTS、记忆管理
FrontEnd/                   桌面窗口加载的前端页面
tools/                      辅助脚本，包括环境安装、轻量分类器构建和 Index-TTS 服务桥接
utils/                      分类器与 RAG 检索模块
assets/classifier/          release 内置轻量路由分类器
tests/                      单元测试
docs/                       说明文档
download_all_models.py      基础模型下载脚本，Index-TTS 可选
train_classifier.py         旧版 BERT 分类器训练脚本，仅开发者需要
tools/build_route_classifier.py  从 classifier_corpus.csv 生成 release 轻量分类器
desktop_launcher.py         桌面启动入口
install.bat                 创建 .venv 并安装依赖
download_models.bat         下载基础模型
start_xinbao_desktop.bat    启动桌面版
```

## 不要上传的内容

请确认以下内容不会进入公开 GitHub 仓库：

- `config.json` 中的真实 API Key；
- `models/` 下的模型本体；
- `chroma_db/` 本地记忆数据库；
- `static/reply_*.mp3` 等运行音频；
- `.venv/`、`.python/`、`.tools/`、`.uv-cache/`、`venv/` 等本地环境和安装缓存；
- 个人报告、查新报告、临时缓存。

## 测试

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

如果还没有安装 `.venv`，也可以在已有 Python 环境中运行同样的 unittest 命令做代码级检查。
