# Index-TTS 可选部署说明

Index-TTS 是芯宝的可选语音增强方案，不是基础安装的必需依赖。

不配置 Index-TTS 时，芯宝默认使用 edge-tts，可以直接运行。

## 1. 下载官方仓库

请先阅读 Index-TTS 官方文档：

<https://github.com/index-tts/index-tts>

建议把官方源码放在不含中文的路径，例如：

```text
D:\IndexTTS\index-tts-src
```

不要把 Index-TTS 官方源码复制到芯宝仓库里。

## 2. 创建独立环境

在 Index-TTS 官方仓库目录中，按照官方文档创建它自己的 `.venv`。

芯宝主环境只负责运行 Flask、记忆、edge-tts 和桌面启动器；Index-TTS 使用自己的环境，通过本机 HTTP 服务提供语音生成。

Windows 首次安装时，建议先按照官方基础流程配置，不要默认安装需要本地编译的可选 `flash-attn` 加速依赖。

## 3. 下载可选模型

先在芯宝项目环境中安装基础依赖，然后在芯宝项目根目录执行：

```powershell
conda activate xinbao_env
python download_all_models.py --with-indextts
```

基础命令：

```powershell
python download_all_models.py
```

只下载 embedding 和 reranker，不下载 Index-TTS。

Index-TTS 模型会放在：

```text
models\IndexTTS\checkpoints
```

`models/` 已加入 `.gitignore`，不要把模型权重上传到 GitHub。

## 4. 准备参考音频

准备一段 5～15 秒的干净女声 WAV，要求：

- 单人说话
- 无背景音乐
- 尽量无混响
- 发音清楚
- 情绪自然

放到：

```text
models\IndexTTS\xinbao_voice.wav
```

## 5. 启动 Index-TTS 服务

使用 Index-TTS 官方环境运行芯宝提供的服务脚本：

```powershell
cd /d D:\IndexTTS\index-tts-src
.venv\Scripts\python.exe "D:\path\to\IntelliChat-Platform\tools\indextts_service.py" --host 127.0.0.1 --port 7862 --indextts-src "D:\IndexTTS\index-tts-src"
```

看到下面的提示，说明服务已经监听：

```text
Index-TTS service listening on http://127.0.0.1:7862
```

## 6. 切换芯宝配置

在本地 `config.json` 的 `voice_settings` 中把引擎改为：

```json
{
  "engine": "indextts",
  "fallback_engine": "edge_tts",
  "indextts_service_url": "http://127.0.0.1:7862/tts",
  "indextts_model_dir": "models\\IndexTTS\\checkpoints",
  "indextts_speaker_audio": "models\\IndexTTS\\xinbao_voice.wav"
}
```

重新启动芯宝桌面版：

```powershell
start_xinbao_desktop.bat
```

如果 Index-TTS 服务未启动、模型缺失、参考音频无效或生成超时，后端会回退到 edge-tts。

## 7. 切回基础路线

如果暂时不想使用 Index-TTS，把配置改回：

```json
{
  "engine": "edge_tts",
  "fallback_engine": "edge_tts"
}
```

此时不需要启动 Index-TTS 服务。
