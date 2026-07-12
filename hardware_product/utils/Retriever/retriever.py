import os
import re
import hashlib

import chromadb


def create_rag_retriever(md_path, embed_model=None, collection=None, top_k=2):
    """
    创建 RAG 检索器。
    - embed_model: 共享的 SentenceTransformer 实例 (避免香橙派上重复加载)
    - collection: 共享的 ChromaDB collection
    """
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    try:
        db_dir = os.path.join(os.path.dirname(md_path), "chroma_db")
        client = chromadb.PersistentClient(path=db_dir)
        collection = client.get_or_create_collection(name="qbit_memory")

        with open(md_path, "r", encoding="utf-8") as f:
            text = f.read()

        raw_blocks = re.split(r"\n(?=#+ )|\n\s*\n", text)
        structured_chunks = []
        current_title = "默认段落"
        chunk_index = 0

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            if block.startswith("#"):
                lines = block.split("\n")
                current_title = lines[0].replace("#", "").strip()
                block = "\n".join(lines[1:]).strip()
            if not block:
                continue

            sentences = re.split(r"([。!?！？])", block)
            sentences = ["".join(i) for i in zip(sentences[0::2], sentences[1::2] + [""])]

            current_chunk_text = ""
            for sentence in sentences:
                if len(current_chunk_text) + len(sentence) > 300:
                    if current_chunk_text.strip():
                        unique_str = f"{os.path.basename(md_path)}_{chunk_index}"
                        chunk_id = hashlib.md5(unique_str.encode("utf-8")).hexdigest()[:12]
                        structured_chunks.append({
                            "id": chunk_id,
                            "text": current_chunk_text.strip(),
                            "metadata": {
                                "type": "knowledge",
                                "source": os.path.basename(md_path),
                                "title": current_title,
                                "chunk_index": chunk_index,
                            },
                        })
                        chunk_index += 1
                        current_chunk_text = ""
                current_chunk_text += sentence

            if current_chunk_text.strip():
                unique_str = f"{os.path.basename(md_path)}_{chunk_index}"
                chunk_id = hashlib.md5(unique_str.encode("utf-8")).hexdigest()[:12]
                structured_chunks.append({
                    "id": chunk_id,
                    "text": current_chunk_text.strip(),
                    "metadata": {
                        "type": "knowledge",
                        "source": os.path.basename(md_path),
                        "title": current_title,
                        "chunk_index": chunk_index,
                    },
                })
                chunk_index += 1

        # 增量更新
        existing_data = collection.get(include=["metadatas"])
        existing_ids = set(existing_data["ids"])
        new_chunks = [c for c in structured_chunks if c["id"] not in existing_ids]

        if new_chunks:
            print(f"🚀 检测到 {len(new_chunks)} 个新知识块，正在进行向量化并写入 ChromaDB...")
            ids = [c["id"] for c in new_chunks]
            docs = [c["text"] for c in new_chunks]
            metas = [c["metadata"] for c in new_chunks]

            # 使用共享的 embedding 模型
            if embed_model is not None:
                embs = embed_model.encode(docs, normalize_embeddings=True).tolist()
            else:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
                embs = _model.encode(docs, normalize_embeddings=True).tolist()

            collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            print("✅ 新知识持久化写入完成!")
        else:
            print("⚡ 知识库无更新，直接从硬盘 ChromaDB 加载，实现秒开!")

        def retrieve(question):
            # 使用共享 embedding 模型
            if embed_model is not None:
                query_embedding = embed_model.encode(
                    question, normalize_embeddings=True
                ).tolist()
            else:
                from sentence_transformers import SentenceTransformer
                _model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
                query_embedding = _model.encode(
                    question, normalize_embeddings=True
                ).tolist()

            results = collection.query(
                query_embeddings=[query_embedding], n_results=top_k
            )

            relevant_contexts = []
            if results["documents"] and len(results["documents"][0]) > 0:
                for i in range(len(results["documents"][0])):
                    doc = results["documents"][0][i]
                    meta = results["metadatas"][0][i]
                    context_str = f"【类型: {meta['type']} | 来源: {meta['source']} | 章节: {meta['title']}】\n{doc}"
                    relevant_contexts.append(context_str)

            if relevant_contexts:
                return "\n\n...\n\n".join(relevant_contexts)
            return "未找到相关背景知识。"

        return retrieve

    except Exception as e:
        print(f"初始化检索器失败: {str(e)}")
        return lambda _: "检索器初始化失败，请检查数据库状态。"
