# ✨ 芯宝 (Xinbao) - 具备长程动态记忆的端云协同陪伴机器人

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Flask-Backend-green)
![Database](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![LLM](https://img.shields.io/badge/DeepSeek-API-black)

## 📖 项目简介

本项目旨在开发一款支持边缘设备部署的智能对话陪伴机器人。系统创新性地采用了**“端云协同与隐私隔离”**的异构计算架构：利用云端大模型（DeepSeek V4）保证极高的认知推理能力，同时将检索增强生成（RAG）管道、用户偏好数据（ChromaDB）100% 本地化部署。

结合创新的**“异步长程记忆引擎”**与**“动态好感度情绪系统”**，机器人能够自动沉淀用户画像，展现出拟生化的情感演化，为用户提供有温度、护隐私的深度情绪陪伴。

---

## 🛠️ 核心架构演进 (Tech Stack)

本项目已从早期的纯本地量化模型（SLM），升级为更符合工业级应用场景的**端云解耦架构**：

- 🧠 **认知与决策大脑 (Cloud-Edge Collaborative)**
  - **云端算力底座：** DeepSeek API (负责深度语义理解与共情回复)
  - **本地长程记忆中枢：** ChromaDB 向量数据库 + 本地轻量级 Embedding 模型 (SentenceTransformer)
  - **双引擎智能路由：** BERT 意图分类器 + 动态启发式唤醒词池 (精准调度闲聊与 RAG 模式)

- 💖 **情感与交互引擎 (Emotion & Interaction)**
  - **多维动态 Prompt 注入：** 根据实时交互动态替换系统级 Prompt，实现世界观“思想钢印”。
  - **好感度奖惩系统：** 启发式规则匹配，动态调整机器人情绪阈值（傲娇/软萌/平静）。

- 🖥️ **可视化交互终端 (Frontend)**
  - **赛博朋克风 Web UI：** 零依赖的 HTML/JS/CSS，内置系统配置面板，支持动态加载 API Key 与初始化主人档案。

- 🔌 **硬件部署规划 (Target Hardware)**
  - 主控平台：Orange Pi 5 / Raspberry Pi 5 等边缘 Linux 单板计算机。
  - 后续规划外设：麦克风阵列 (ASR)、扬声器 (TTS)。

---

```python
markdown_content = """# ✨ 芯宝 (Xinbao) - 具备长程动态记忆的端云协同情感陪伴机器人

[![Python Version](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Flask-Backend-green?style=flat-square&logo=flask)](https://flask.palletsprojects.com/)
[![Database](https://img.shields.io/badge/ChromaDB-Vector_Store-orange?style=flat-square)](https://www.trychroma.com/)
[![CUDA Version](https://img.shields.io/badge/CUDA-12.8--cu128-76B900?style=flat-square&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Target Platform](https://img.shields.io/badge/Hardware-OrangePi%205%20%7C%20PC-red?style=flat-square)](http://www.orangepi.org/)

本项目是一项**省级大学生创新创业训练计划（大创）**的核心成果。针对市面上情感陪伴机器人存在的隐私泄露、记忆碎片化、情感表达生硬等痛点，我们设计并实现了一款支持端侧（边缘设备）轻量化部署、具备长程动态记忆与拟生化流式状态机的端云协同陪伴机器人——**芯宝 (Xinbao)**。

---

## 💡 核心设计理念与技术创新

### 1. 端云协同与隐私隔离架构 (Edge-Cloud Hybrid)
为了在保障核心用户隐私的同时获得顶级的认知与共情能力，系统采用创新的**异构双层架构**：
- **云端大脑：** 统一采用云端大模型（如 DeepSeek-Chat）作为共情文本生成引擎，输出高质量、带情绪倾向的陪伴话术。
- **本地芯片：** 用户偏好、长期生活记忆、人设资料 100% 留存在本地边缘设备，通过本地向量管道完成检索增强生成（RAG），云端无法获取用户的完整隐私画像。

### 2. 双引擎智能路由中枢 (Intelligent Routing)
后端内置智能意图路由中枢，打破了传统 RAG 盲目检索造成的延迟与上下文污染：
- **规则拦截机制：** 维护一套高动态的启发式唤醒词池（包含静态设定的核心实体与大模型自主学习提炼的唤醒词），精准拦截特定的私人或学术提问。
- **分类器深度判定：** 未命中规则的语料由本地微调训练的 **BERT 意图分类器** 进行实时推断，实现“直接闲聊生成”与“知识库检索增强”的双路毫秒级并轨调度。

### 3. 双重质检检索漏斗 (Recall & Rerank Funnel)
为了解决海量记忆长河中向量空间拥挤、字面不匹配带来的检索降级问题，系统构建了双层过滤漏斗：
- **阶段一：向量粗排 (Vector Recall)**
  通过本地挂载的轻量级向量模型（`SentenceTransformer`）将问题升维，在本地 `ChromaDB` 多维空间中放宽检索标准快速捞取 Top-10 候选记忆。
- **阶段二：交叉注意力精排 (Cross-Attention Reranking)**
  利用本地高精度的 **BGE Reranker (CrossEncoder)** 精排引擎进行细粒度语义交叉比对，实行正分截断拦截（Score > 0），最多仅提炼前 3 条置信度最高的私人动态记忆喂给大模型，彻底消除大模型关于主人记忆的“幻觉”。

### 4. 异步长程记忆沉淀引擎 (Async Memory Engine)
系统具备拟生化的“边聊边记”能力。在每一轮流式交互的最终阶段，系统会通过多线程管道启动一个**异步记忆提取器**：
- 调用大模型分析本次对话是否蕴含长期价值信息。
- 自动剥离无用废话，将其提炼为第三人称客观陈述句，盖上**精准的时间戳钢印**。
- 自动提炼专属唤醒词追加到本地，并将记忆持久化写入物理芯片。在后续检索中，机器人能够对比当前时间锚点完成敏捷的时序逻辑推理与溯源引用。

### 5. 情感演化与实时流式颜艺系统 (Emotion Stream Mechanism)
拒绝生硬的机械式对答，让机器人具备真正的动态性格：
- **好感度持久化账本：** 内置启发式奖惩规则，根据用户的赞赏（如“真棒”、“贴贴”）或苛责（如“笨”、“很烦”）实时加减好感度，并锁死在 `[0, 100]` 的区间内。
- **情绪驱动状态机 (Mood Machine)：** 好感度直接决定机器人的四阶情绪（傲娇、受气委屈、平静阳光、软萌粘人），动态重写系统级 Prompt 思想钢印。
- **首包变脸推送 (Early-Response Expression)：** 在后端向前端吐出任何模型文本碎块之前，系统率先通过 `Server-Sent Events (SSE)` 流通道向前端空投好感度变脸信号，使机器人的虚拟面部（`happy`/`sad`/`shy` 类名切换）先于文字和语音发生变化，带来即时、顺畅的感官反馈。

### 6. 多线程安全语音发声管道 (Multi-Threaded TTS Pipeline)
针对 Waitress 生产级多线程容器的并发需求，后端重构了异步执行管道：
- 净化大模型回复中的 Markdown 标识符、粗体框及 Emoji，提炼纯净文本。
- 采用面向生产环境的 `asyncio.new_event_loop()` 隔离机制，驱动 `Edge-TTS` 引擎高速同步生成音频。
- **自动清理机制：** 采用毫秒级时序轮询，自动销毁 `static/` 目录下超过 3 分钟的旧 MP3 音频，防止端侧设备存储器爆炸。
- **前端静音解锁术：** 前端在用户发送消息的第一时间，以 `volume = 0` 静音播放 Base64 空白音频，骗过并攻破现代浏览器极其严苛的自动播放安全拦截策略，确保最终动态音频（`audio_url`）传回时实现 100% 自动发声。

---

## 📂 项目目录结构


```

```text
SUCCESS: README.md generated.

```text
Your-Desktop-dialogue-robot/
├── BackEnd/
│   └── simple.py               # 生产级后端认知引擎 (Waitress 容器 + 流式响应 + TTS + 路由)
├── FrontEnd/
│   └── robot.html              # 赛博朋克风格 Web UI 终端 (零依赖 + CRT特效 + 实时变脸状态机)
├── Logs/
│   └── log_5_2.md              # 研发迭代日志与技术沉淀
├── models/
│   ├── classifier/             # 本地 BERT 意图分类器权重目录
│   ├── embedding/              # 本地 SentenceTransformer 粗排向量模型
│   └── reranker/               # 本地 BGE CrossEncoder 交叉注意力精排模型
├── static/                     # 🔴 实时语音流临时缓冲区 (需手动创建或靠代码保障)
├── uploads/                    # 本地多模态文件上传暂存区
├── utils/
│   ├── Classifier/             # 分类器核心算法与数据增强组件
│   └── Retriever/              # 本地 RAG 结构知识库检索器
├── classifier_corpus.csv       # 意图分类器本地训练语料集
├── config.json                 # 系统核心物理配置文件 (包含人设、接口、路径、动态配置)
├── download_all_models.py      # 模型自动化拉取初始化脚本
├── dynamic_keywords.txt        # 边缘设备自主学习提炼的唤醒词账本
├── knowledge.md                # 芯宝静态世界观与结构化知识库
├── models_ready.bat            # 一键环境初始化批处理脚本 (连环触发下载与分类器微调)
├── requirements.txt            # 1:1 完美复刻纯净测试环境的物理依赖清单
└── train_classifier.py         # 本地分类器微调训练入口脚本

```

---

## 🚀 快速开始 (Quick Start)

本项目已全面通过纯净环境下的全链路闭环联调测试。请遵循以下标准流程进行 PC 端的仿真运行：

### 1. 克隆代码库与初始化静态目录

首先，将项目代码克隆到本地机器上：

```bash
git clone [https://github.com/YourUsername/Your-Desktop-dialogue-robot.git](https://github.com/YourUsername/Your-Desktop-dialogue-robot.git)
cd Your-Desktop-dialogue-robot

```

**🚨 核心注意项：** 请确保项目根目录下存在名为 `static` 的文件夹（通常已包含在仓库中，如没有请手动创建）。该文件夹用于提供本地 TTS 音频流的动态擦写服务：

```bash
mkdir static

```

### 2. 配置纯净运行环境

为了防止不同项目间深度学习框架版本的严重冲突，强烈建议使用 Anaconda 隔离出一个全新的虚拟环境：

```bash
conda create -n xinbao_env python=3.10
conda activate xinbao_env

```

### 3. 安装显卡加速版 PyTorch 深度学习底座

**特别提醒：** 本系统完美匹配新一代显卡架构（如 **NVIDIA GeForce RTX 5060 Laptop GPU**）。为了防止系统被默认装上残血的 CPU 版本，或者遇到经典的新硬件撞上老软件抛出 `RuntimeError: CUDA error: no kernel image is available for execution on the device` 错误，**请务必针对最新显卡架构及 CUDA 12.8 驱动环境安装满血加速版 PyTorch**：

```bash
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu128](https://download.pytorch.org/whl/cu128)

```

### 4. 安装基础生产依赖

依赖项中包含了向量处理、多线程服务器、通信模块等。在虚拟环境中一键运行：

```bash
pip install -r requirements.txt

```

### 5. 激活本地双层模型引擎 (One-Click Initialization)

双击运行根目录下的自动化批处理脚本：

```bash
models_ready.bat

```

或者在终端中手动触发：

```bash
python download_all_models.py
python train_classifier.py

```

系统将自动检测并下载所需的几百兆向量模型、精排模型，并就地提取 `classifier_corpus.csv` 语料集完成分类器的本地微调训练。当控制台打出 `[✅ 训练完成] 分类器权重已稳固导出！` 即可。

### 6. 点火运行后端中枢

在确保依赖与模型全部就位后，执行路径拉起 Flask 服务：

```bash
python BackEnd/simple.py

```

系统将基于生产级安全容器 `Waitress` 启动 4 个高并发并发线程，并在本地 `http://0.0.0.0:5000` 展开高敏捷监听。

### 7. 唤醒前端赛博交互终端

保持后端黑框终端常驻运行，定位至 `FrontEnd/robot.html`：

* **方案 A (推荐)：** 使用 VS Code 的 **Live Server** 插件右键启动，通过 `http://localhost:5500` 打开。这能完美将页面置于合法的网络协议沙盒下，规避高版本浏览器对本地文件协议（`file:///`）施加的极其严格的反跨域安全拦截，让语音流瞬间发声。
* **方案 B (仿真展演)：** 亦可在终端运行 `python -m http.server 8000`，在浏览器访问 `http://localhost:8000/FrontEnd/robot.html`。

### 8. 绑定云端大脑秘钥

在打开的赛博朋克风面板中：

1. 点击左下角的 **`[ SYSTEM CONFIG ]`**。
2. 在弹出的高科技初始化设定框中，填写主人的称呼（如：阿顺）、当前的身份以及近期遇到的压力和状态。
3. 输入你的云端超级大脑秘钥：`DEEPSEEK_API_KEY`（支持流式响应的标准 Key）。
4. 点击 **`SAVE & BOOT`**。前端将设定打包持久化传给本地硬盘 `config.json`，同时芯宝的双眼亮起，正式唤醒！

---

## 🎯 核心交互与表情测试指南

你可以通过以下特定的指令和话术，全面测试和验证系统的深度情感演化功能：

| 用户输入话术示例 | 系统路由判定 | 实时变脸信号 | 芯宝虚拟面部特效表现 | 情感与语音表现 |
| --- | --- | --- | --- | --- |
| **“点击机器人脑袋/身体”** | 前端原生拦截 | `triggerRobotShy()` | 🔴 **害羞状态 (Shy)**：腮红放大 1.5 倍变成深粉红，眼睛变成眯眯眼 | 弹出可爱气泡：“主人~ 芯宝在呢！(///▽///)”，语音同步温柔回应。 |
| **“芯宝好乖/超可爱，摸摸头”** | 闲聊模式（触发奖励词） | `change: "up"` | 🟢 **开心状态 (Happy)**：眼睛变成上扬的弯弯笑眼，腮红亮度拉满 | 后端好感度物理账本 **+3** 分。气泡转换为大字报喜，语气变得极其软萌、撒娇、粘人。 |
| **“你这个小机器人真笨，很烦”** | 闲聊模式（触发惩罚词） | `change: "down"` | 🔵 **沮丧状态 (Sad)**：眼睛变成向下垂的平眼，腮红全部褪去，脑袋边框变暗淡 | 后端好感度物理账本 **-5** 分。气泡转换为委屈痛哭，带 `💢` 标志，说话带有傲娇和轻微小情绪。 |
| **“查询当前好感度/你有多喜欢我”** | 拦截器直接接管 (0 Token) | `none`（保持当前） | 维持当前基本眨眼动画（根据当前所处好感度区间维持不同表情基调） | **不消耗云端大模型 API**。后端直接从持久化 json 中读取分数进行个性化分级回复（如：>80 分求贴贴抱抱；<30 分哼生闷气）。 |
| **“记得我上周去吃了什么吗？”** | **RAG 强行拦截并切换模式** | 根据回复的情感词判定 | 粗排捞取10条 $\rightarrow$ 精排交叉打分过滤 $\rightarrow$ 截断正分语料 | 准确抓取带有精确时间戳的本地 ChromaDB 私人记忆，并在相关句子末尾强制加上类似 `^[来源：YYYY-MM-DD]` 的**学术级引文脚注**。 |

---

## 👥 大创项目团队与致谢

* **指导老师：** 感谢实验室导师为本项目提供的高性能算力支持。
* **核心开发团队（排名不分先后）：** 王勇顺、阳泽怡、徐启恒、杨赛宇、徐语乐。
* **致谢：** 感谢所有团队成员在分类器数据增强、向量数据管道以及前端赛博 UI 的无数次深夜联调与攻坚。

---

## 📜 许可证 (License)

本项目基于 [MIT License](https://www.google.com/search?q=LICENSE) 开源协议。允许个人、学术团队及大创评委自由下载、复现、修改及路演展示。在实体硬件（如树莓派/香橙派）部署时，请务必保留原作者团队的署名。
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(markdown_content)

print("SUCCESS: README.md generated.")
