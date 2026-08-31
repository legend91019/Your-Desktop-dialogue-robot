#!/bin/bash
# ============================================================
# 芯宝 (Xinbao) - 香橙派 AIPro 一键部署脚本
# 使用方法: chmod +x setup_pi.sh && ./setup_pi.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f "$SCRIPT_DIR/config.json" ]; then
    cp "$SCRIPT_DIR/config.example.json" "$SCRIPT_DIR/config.json"
    echo "已从 config.example.json 创建本机配置"
fi

echo "╔══════════════════════════════════════════════╗"
echo "║     芯宝 Xinbao - 香橙派 AIPro 部署脚本       ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ---- 1. 检查 Python ----
echo "[1/5] 检查 Python 环境..."
if ! command -v python3 &>/dev/null; then
    echo "❌ 未找到 python3，请先安装 Python 3.9+"
    echo "   sudo apt update && sudo apt install python3 python3-pip python3-venv -y"
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "   Python 版本: $PYTHON_VERSION"

# ---- 2. 创建虚拟环境 ----
echo ""
echo "[2/5] 创建虚拟环境..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ 虚拟环境创建完成"
else
    echo "   ⚡ 虚拟环境已存在，跳过"
fi

source venv/bin/activate

# ---- 3. 安装依赖 ----
echo ""
echo "[3/5] 安装 Python 依赖 (这可能需要几分钟)..."
pip install --upgrade pip -q

# 先装 PyTorch CPU 版
echo "   ⏳ 安装 PyTorch (CPU 版, ARM64)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q

# 再装其余依赖
echo "   ⏳ 安装其余依赖..."
pip install -r requirements_pi.txt -q

echo "   ✅ 依赖安装完成"

# ---- 4. 下载模型 ----
echo ""
echo "[4/5] 下载模型文件 (从魔搭镜像)..."
MODELS_DIR="$SCRIPT_DIR/models"

if [ ! -d "$MODELS_DIR/reranker" ] || [ ! -d "$MODELS_DIR/embedding" ]; then
    python3 -c "
import os
from modelscope.hub.snapshot_download import snapshot_download

models_dir = '$MODELS_DIR'
os.makedirs(models_dir, exist_ok=True)

# 1. 分类器 (BERT 中文) - 首次运行时会自动下载
print('📦 分类器 (bert-base-chinese) 将在首次启动时自动下载到 models/classifier')

# 2. 精排模型
print('⏳ 下载 BGE 精排模型...')
snapshot_download('Xorbits/bge-reranker-base', cache_dir=os.path.join(models_dir, 'reranker'))
print('✅ 精排模型完成')

# 3. 向量化模型
print('⏳ 下载 BGE 向量化模型...')
snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir=os.path.join(models_dir, 'embedding'))
print('✅ 向量化模型完成')
"
    echo "   ✅ 模型下载完成"
else
    echo "   ⚡ 模型目录已存在，跳过下载"
fi

# ---- 5. 创建必要目录 ----
echo ""
echo "[5/5] 创建运行时目录..."
mkdir -p "$SCRIPT_DIR/static" "$SCRIPT_DIR/uploads" "$SCRIPT_DIR/chroma_db"
touch "$SCRIPT_DIR/dynamic_keywords.txt"

# ---- 完成 ----
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║            ✅ 部署完成!                        ║"
echo "╠══════════════════════════════════════════════╣"
echo "║  启动方式:                                    ║"
echo "║    1. 手动: source venv/bin/activate          ║"
echo "║             python3 app/main.py               ║"
echo "║    2. 后台: bash run.sh                       ║"
echo "║    3. 开机自启: sudo bash install_service.sh   ║"
echo "║                                                ║"
echo "║  访问地址: http://<香橙派IP>:5000               ║"
echo "║  先在设置里填入 DeepSeek API Key!               ║"
echo "╚══════════════════════════════════════════════╝"
