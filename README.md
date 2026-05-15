# ✨ 芯宝 (Xinbao) - 具备长程动态记忆的端云协同陪伴机器人

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![Framework](https://img.shields.io/badge/Flask-Backend-green)
![Database](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![LLM](https://img.shields.io/badge/DeepSeek-API-black)

## 📖 项目简介

本项目旨在开发一款支持边缘设备部署的智能对话陪伴机器人。系统创新性地采用了**“端云协同与隐私隔离”**的异构计算架构：利用云端大模型（DeepSeek V3）保证极高的认知推理能力，同时将检索增强生成（RAG）管道、用户偏好数据（ChromaDB）100% 本地化部署。

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

## 🚀 快速开始 (Quick Start - PC 仿真与联调阶段)

目前项目已跑通全链路软件闭环，您可以通过以下步骤在本地启动“意图仲裁 + RAG + 大模型”的完整测试：

### 1. 环境准备
- 安装 Python 3.9 或更高版本。
- 建议使用 `conda` 或 `venv` 创建虚拟环境。

### 2. 安装核心依赖
在项目根目录下，执行以下命令安装所需的 Python 包：
```bash
pip install flask flask-cors requests chromadb sentence-transformers scikit-learn
