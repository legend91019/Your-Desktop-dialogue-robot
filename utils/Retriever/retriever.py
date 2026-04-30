import os
import chromadb
from sentence_transformers import SentenceTransformer
import re
import hashlib

def create_rag_retriever(md_path: str, model_name: str = "BAAI/bge-small-zh-v1.5", top_k: int = 2) -> callable:
    # 压制无用的警告
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    try:
        # ==================== 1. 初始化 ChromaDB (持久化引擎) ====================
        # 在项目根目录下自动创建一个 chroma_db 文件夹用来存放数据库文件
        db_dir = os.path.join(os.path.dirname(md_path), "chroma_db")
        client = chromadb.PersistentClient(path=db_dir)
        
        # 获取或创建一个名为 qbit_memory 的集合（表）
        collection = client.get_or_create_collection(name="qbit_memory")
        
        # ==================== 2. 读取并结构化数据 (复用上一版的优秀逻辑) ====================
        with open(md_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        raw_blocks = re.split(r'\n(?=#+ )|\n\s*\n', text)
        structured_chunks = []
        current_title = "默认段落"
        chunk_index = 0

        for block in raw_blocks:
            block = block.strip()
            if not block: continue
            
            if block.startswith('#'):
                lines = block.split('\n')
                current_title = lines[0].replace('#', '').strip()
                block = '\n'.join(lines[1:]).strip()
            if not block: continue

            sentences = re.split(r'([。！？])', block)
            sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2] + [""])]
            
            current_chunk_text = ""
            for sentence in sentences:
                if len(current_chunk_text) + len(sentence) > 300:
                    if current_chunk_text.strip():
                        unique_str = f"{os.path.basename(md_path)}_{chunk_index}"
                        chunk_id = hashlib.md5(unique_str.encode('utf-8')).hexdigest()[:12]
                        structured_chunks.append({
                            "id": chunk_id, "text": current_chunk_text.strip(),
                            "metadata": {"type": "knowledge", "source": os.path.basename(md_path), "title": current_title, "chunk_index": chunk_index}
                        })
                        chunk_index += 1; current_chunk_text = ""
                current_chunk_text += sentence

            if current_chunk_text.strip():
                unique_str = f"{os.path.basename(md_path)}_{chunk_index}"
                chunk_id = hashlib.md5(unique_str.encode('utf-8')).hexdigest()[:12]
                structured_chunks.append({
                    "id": chunk_id, "text": current_chunk_text.strip(),
                    "metadata": {"type": "knowledge", "source": os.path.basename(md_path), "title": current_title, "chunk_index": chunk_index}
                })
                chunk_index += 1

        # ==================== 3. 智能增量更新 (ChromaDB 的威力) ====================
        # 去数据库里查一下，现在已经存了哪些 ID
        existing_data = collection.get(include=["metadatas"])
        existing_ids = set(existing_data["ids"])
        
        # 挑出那些数据库里没有的“新数据块”
        new_chunks = [c for c in structured_chunks if c["id"] not in existing_ids]
        
        model = SentenceTransformer(model_name)
        
        if new_chunks:
            print(f"🚀 检测到 {len(new_chunks)} 个新知识块，正在进行向量化并写入 ChromaDB...")
            ids = [c["id"] for c in new_chunks]
            docs = [c["text"] for c in new_chunks]
            metas = [c["metadata"] for c in new_chunks]
            
            # 只有新数据才需要耗时的向量计算！
            embs = model.encode(docs, normalize_embeddings=True).tolist()
            
            # 写入数据库并保存到硬盘
            collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            print("✅ 新知识持久化写入完成！")
        else:
            print("⚡ 知识库无更新，直接从硬盘 ChromaDB 加载，实现秒开！")

        # ==================== 4. 返回检索函数 ====================
        def retrieve(question: str) -> str:
            # 把用户的问题变成向量
            query_embedding = model.encode(question, normalize_embeddings=True).tolist()
            
            # 直接调用 ChromaDB 的原生查询
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                # 以后甚至可以在这里加上 where={"type": "knowledge"} 进行精确过滤
            )
            
            relevant_contexts = []
            # 解析 ChromaDB 的返回格式
            if results['documents'] and len(results['documents'][0]) > 0:
                for i in range(len(results['documents'][0])):
                    doc = results['documents'][0][i]
                    meta = results['metadatas'][0][i]
                    context_str = f"【类型: {meta['type']} | 来源: {meta['source']} | 章节: {meta['title']}】\n{doc}"
                    relevant_contexts.append(context_str)
            
            if relevant_contexts:
                return "\n\n...\n\n".join(relevant_contexts)
            return "未找到相关背景知识。"
            
        return retrieve
        
    except Exception as e:
        print(f"初始化检索器失败: {str(e)}")
        return lambda _: "检索器初始化失败，请检查数据库状态。"