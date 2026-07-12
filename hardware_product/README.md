# 芯宝 Xinbao - 香橙派 AIPro 部署版

端云协同陪伴机器人，适配 ARM Linux (Orange Pi AIPro / Raspberry Pi 5)。

## 快速开始

```bash
# 1. 确保板子上已安装 Python 3.9+
python3 --version

# 2. 克隆或拷贝整个 hardware_product 文件夹到板子上
scp -r hardware_product/ pi@<香橙派IP>:/home/pi/xinbao/

# 3. 一键部署
cd /home/pi/xinbao
chmod +x setup_pi.sh
./setup_pi.sh

# 4. 启动
bash run.sh start

# 5. 浏览器访问
# http://<香橙派IP>:5000
```

## 首次使用

1. 打开浏览器访问 `http://<香橙派IP>:5000`
2. 点击 `[ SYSTEM CONFIG ]`
3. 填入你的 **DeepSeek API Key** 和主人信息
4. 点击 `SAVE & BOOT` 开始对话

## 文件结构

```
hardware_product/
├── app/main.py              # Flask 后端主程序
├── app/tools/time_tool.py   # 时间工具
├── utils/                   # 工具模块 (分类器、检索器)
├── frontend/robot.html      # 赛博朋克风 Web UI
├── config.json              # 配置文件
├── knowledge.md             # 知识库 (静态设定 & 世界观)
├── requirements_pi.txt      # ARM Linux 依赖
├── setup_pi.sh              # 一键部署脚本
├── run.sh                   # 启动/停止脚本
├── install_service.sh       # 安装 systemd 开机自启
└── xinbao.service           # systemd 服务模板
```

## 管理命令

```bash
bash run.sh start      # 启动
bash run.sh stop       # 停止
bash run.sh status     # 查看状态
bash run.sh restart    # 重启

# 开机自启
sudo bash install_service.sh
```

## 模型说明

| 模型 | 路径 | 用途 | 大小 |
|------|------|------|------|
| BERT 分类器 | `models/classifier/` | 意图识别 (闲聊/RAG) | ~400MB |
| BGE Embedding | `models/embedding/` | 文本向量化 | ~100MB |
| BGE Reranker | `models/reranker/` | 记忆精排 | ~1GB |

模型通过 `setup_pi.sh` 从魔搭下载。也可从 PC 拷贝已有模型到 `models/` 目录。

## 架构

```
[浏览器] <--> [Flask :5000] <--> [DeepSeek API (云端)]
                    |
            本地推理引擎:
            ├── BERT 分类器 (CPU)
            ├── ChromaDB (长期记忆)
            ├── BGE Embedding (向量化)
            ├── BGE Reranker (记忆精排)
            └── Edge-TTS (语音合成)
```

**大模型推理全部走云端**，香橙派只跑轻量级 BERT + 向量检索，CPU 完全够用。
