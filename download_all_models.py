import os
from modelscope.hub.snapshot_download import snapshot_download

# 获取项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

print("📦 开始执行中心化模型资产下载协议...")

# 1. 下载 Reranker (精排模型)
reranker_local_path = os.path.join(MODELS_DIR, "reranker")
print("\n⏳ [1/2] 正在下载 BGE 精排模型 (bge-reranker-base) 至 models/reranker ...")
# 从魔搭高速下载精排模型
snapshot_download('Xorbits/bge-reranker-base', cache_dir=reranker_local_path, local_dir=reranker_local_path)
print("✅ 精排模型下载完成！")

# 2. 下载 Embedding (向量化模型) - 对应你 config 里的 bge-small-zh-v1.5
embedding_local_path = os.path.join(MODELS_DIR, "embedding")
print("\n⏳ [2/2] 正在下载 BGE 向量化模型 (bge-small-zh-v1.5) 至 models/embedding ...")
# 从魔搭高速下载向量化模型
snapshot_download('AI-ModelScope/bge-small-zh-v1.5', cache_dir=embedding_local_path, local_dir=embedding_local_path)
print("✅ 向量化模型下载完成！")

print("\n🎉 所有动态模型资产已安全沉淀到本地硬盘 (models/ 文件夹)！")