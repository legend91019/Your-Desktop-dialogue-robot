<p align="center">
  <img src="assets/branding/xinbao-logo.jpg" alt="芯宝 Xinbao logo" width="260">
</p>

<h1 align="center">芯宝 Xinbao</h1>

<p align="center">面向 Windows 的本地记忆对话陪伴机器人</p>

<p align="center">厦门大学省级大创项目 · CRAIC 中国机器人及人工智能大赛福建赛区一等奖 · 国家级一等奖</p>

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

## 下载与使用：两条路径

请先区分“普通用户安装器”和“GitHub 源码包”。它们的操作方式不同：

| 适用人群 | 下载内容 | 能否双击即用 | 需要执行脚本 |
| --- | --- | --- | --- |
| 普通用户（推荐） | Release 说明中的 `Xinbao-Setup-v<version>.exe` | 可以，运行安装器后从桌面快捷方式启动 | 不需要 |
| 开发者/需要查看代码的人 | GitHub Release 的 `Source code (zip)` | 不可以，这是源码包 | 需要 |

### 路径 A：普通用户下载 EXE

完整 Windows 安装器包含私有 Python 运行环境、CUDA PyTorch、本地模型和芯宝程序。当前安装器约 3.55 GiB，超过 GitHub Release 单个资产的 2 GiB 限制，因此请从 [v1.0.7 Release 说明](https://github.com/legend91019/Your-Desktop-dialogue-robot/releases/tag/v1.0.7)中的夸克网盘链接下载，也可以直接使用下面的地址：

- 夸克网盘：[下载 Xinbao](https://pan.quark.cn/s/4a4913e45811?pwd=tpCL)
- 提取码：`tpCL`
- 文件名：`Xinbao-Setup-v1.0.7.exe`

1. 下载 `Xinbao-Setup-v1.0.7.exe`。
2. 双击 EXE，按安装向导选择安装位置并完成安装。
3. 从桌面或开始菜单打开“芯宝 Xinbao”。这个 EXE 是安装器，安装完成后再从快捷方式启动程序；不需要安装 Conda、Python、pip，也不需要运行 `install.bat`、`download_models.bat` 或其他脚本。
4. 首次打开后，在设置面板填写自己的 DeepSeek API Key 和主人信息并保存。

可选的文件校验（PowerShell）：

```powershell
Get-FileHash .\Xinbao-Setup-v1.0.7.exe -Algorithm SHA256
```

将结果与 [v1.0.7 Release 说明](https://github.com/legend91019/Your-Desktop-dialogue-robot/releases/tag/v1.0.7)中的 SHA-256 值比较。安装器文件名中的版本号会随发布版本变化。

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

### 路径 B：从 GitHub Release 下载源码

GitHub Release 中的 `Source code (zip)` 只包含项目代码和轻量资源，不包含 `.venv`、模型权重、个人记忆或构建产物。因此源码包不能直接双击打开，适合开发、调试和二次修改。

1. 安装 CPython 3.10 或 3.11，并确保电脑已安装支持 CUDA 12.x 的 NVIDIA 驱动。
2. 解压源码包，在项目根目录打开命令提示符（该目录应能看到 `install.bat`）。
3. 依次运行以下三个脚本：

```bat
install.bat
download_models.bat
start_xinbao_desktop.bat
```

首次运行的作用分别是：创建项目私有 `.venv` 并安装依赖、下载基础模型、启动桌面窗口。之后再次使用时只需运行 `start_xinbao_desktop.bat`。

需要 Index-TTS 开发资源时，可在项目根目录运行 `python tools/download_all_models.py --with-indextts`；普通用户安装器已提供默认的 edge-tts 语音路径。

这些脚本只服务于开发和调试，不是普通用户的安装步骤。构建发布运行时使用项目私有 `.venv`，不读取 Conda 环境；`hardware_product` 保留为硬件版本目录，不属于当前桌面版发布范围。

运行测试：

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

## 项目结构

```text
.
├─ BackEnd/                       Flask/Waitress 后端
│  ├─ simple.py                   对话接口、RAG 调用和后端路由
│  ├─ memory_admin.py             长期记忆的管理接口
│  ├─ tts_engine.py               edge-tts / Index-TTS 语音适配
│  ├─ audio_player.py             本地音频播放
│  └─ tools/time_tool.py          时间工具
├─ FrontEnd/robot.html             桌面窗口加载的聊天页面
├─ utils/
│  ├─ Retriever/retriever.py       文档切分、向量检索和重排
│  └─ Classifier/                  对话路由分类器
├─ assets/
│  ├─ branding/                    芯宝 logo
│  └─ classifier/                  发布版轻量分类器文件
├─ tools/                          发布构建、环境和模型辅助脚本
│  ├─ bootstrap_release.ps1        准备发布环境
│  ├─ build_runtime.ps1            构建私有运行时
│  ├─ build_installer.ps1          构建 Windows 安装器
│  └─ create_model_manifest.py     生成模型校验清单
├─ desktop_launcher.py             启动后端并创建桌面窗口
├─ runtime_paths.py                程序资源和 %APPDATA% 路径管理
├─ startup_checks.py               GPU、模型、校验值和端口检查
├─ download_all_models.py          基础模型下载入口
├─ install.bat                     源码用户安装依赖
├─ download_models.bat             源码用户下载模型
├─ start_xinbao_desktop.bat        源码用户启动桌面端
├─ packaging/                      Inno Setup 配置和 Windows 图标
├─ tests/                           自动化测试
├─ docs/                            用户指南、版本说明和发布清单
└─ hardware_product/               香橙派硬件版本，不参与 Windows 桌面发布
```

桌面端启动链路如下：`desktop_launcher.py` 先执行 `startup_checks.py`，找到本机可用端口后启动 `BackEnd.simple`；后端就绪后，使用 pywebview 打开 `FrontEnd/robot.html`。程序文件安装在安装目录，配置、记忆和运行日志写入 `%APPDATA%\Xinbao\`，两者相互分离。

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
