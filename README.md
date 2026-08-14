# 芯宝 Xinbao：桌面对话陪伴机器人

芯宝是一个面向桌面演示与后续边缘设备部署的对话机器人项目。当前版本以 PC 桌面端为主，提供：

- DeepSeek 云端对话能力；
- 本地 ChromaDB 长程记忆与 RAG 检索；
- 可管理记忆的后端接口；
- 桌面窗口启动器；
- 默认 `edge-tts` 语音输出；
- 可选 Index-TTS 高音质语音方案。

本仓库的公开版本不应包含 API Key、本地数据库、运行音频、模型权重或个人报告材料。

当前版本：`v1.0.1`。这是对 `v1.0.0` 的安装器修复版，重点解决 release 测试中暴露的系统 Python 3.13 与 CUDA PyTorch 依赖安装问题。

## 快速开始

普通用户按这三个脚本走即可：

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

安装器会优先寻找 Python 3.11 / 3.10。如果电脑上只有 Python 3.13，安装器会停止并提示安装兼容版本，避免依赖装到一半才失败。

默认安装 CUDA 12.8 版 PyTorch。纯 CPU 电脑可以使用：

```bat
python tools\setup_env.py --torch cpu
```

### 2. 下载基础模型

双击运行：

```bat
download_models.bat
```

该脚本会下载基础 RAG / 向量检索模型，并训练本地分类器。它不会下载 Index-TTS 大模型。

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
tools/                      辅助脚本，包括环境安装和 Index-TTS 服务桥接
utils/                      分类器与 RAG 检索模块
tests/                      单元测试
docs/                       说明文档
download_all_models.py      基础模型下载脚本，Index-TTS 可选
train_classifier.py         本地分类器训练脚本
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
- `.venv/`、`venv/` 等虚拟环境；
- 个人报告、查新报告、临时缓存。

## 测试

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

如果还没有安装 `.venv`，也可以在已有 Python 环境中运行同样的 unittest 命令做代码级检查。
