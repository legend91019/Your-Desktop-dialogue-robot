import hashlib


def _vector_for_text(embed_model, text):
    encoded = embed_model.encode([text], normalize_embeddings=True)
    if hasattr(encoded, "tolist"):
        return encoded.tolist()[0]
    return encoded[0]


def _normalize_memory(memory_id, document, metadata):
    metadata = metadata or {}
    return {
        "id": memory_id,
        "text": document or "",
        "timestamp": metadata.get("timestamp", ""),
        "status": metadata.get("status", "approved"),
        "source": metadata.get("source", "dynamic_memory"),
        "metadata": metadata,
    }


def _get_memory_metadata(collection, memory_id):
    try:
        data = collection.get(ids=[memory_id], include=["metadatas"])
    except TypeError:
        data = collection.get(include=["metadatas"])

    ids = data.get("ids") or []
    metadatas = data.get("metadatas") or []
    if memory_id not in ids:
        return {}
    return metadatas[ids.index(memory_id)] or {}


def list_memories(collection):
    data = collection.get(
        where={"source": "dynamic_memory"},
        include=["documents", "metadatas"],
    )
    ids = data.get("ids") or []
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []

    memories = [
        _normalize_memory(memory_id, document, metadata)
        for memory_id, document, metadata in zip(ids, documents, metadatas)
    ]
    return sorted(memories, key=lambda item: item.get("timestamp") or "", reverse=True)


def add_memory(collection, embed_model, text, timestamp):
    text = (text or "").strip()
    if not text:
        raise ValueError("memory text cannot be empty")

    memory_id = hashlib.md5(text.encode("utf-8")).hexdigest()[:12]
    metadata = {
        "type": "user_preference",
        "source": "dynamic_memory",
        "timestamp": timestamp,
        "title": "主人动态画像",
        "chunk_index": 9999,
        "status": "approved",
        "origin": "manual",
    }
    embedding = _vector_for_text(embed_model, text)
    collection.upsert(
        ids=[memory_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    return _normalize_memory(memory_id, text, metadata)


def update_memory(collection, embed_model, memory_id, text):
    memory_id = (memory_id or "").strip()
    text = (text or "").strip()
    if not memory_id:
        raise ValueError("memory id cannot be empty")
    if not text:
        raise ValueError("memory text cannot be empty")

    metadata = _get_memory_metadata(collection, memory_id)
    metadata.update({
        "type": metadata.get("type", "user_preference"),
        "source": "dynamic_memory",
        "title": metadata.get("title", "主人动态画像"),
        "chunk_index": metadata.get("chunk_index", 9999),
        "status": metadata.get("status", "approved"),
    })
    embedding = _vector_for_text(embed_model, text)
    collection.upsert(
        ids=[memory_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
    )
    return _normalize_memory(memory_id, text, metadata)


def delete_memory(collection, memory_id):
    memory_id = (memory_id or "").strip()
    if not memory_id:
        raise ValueError("memory id cannot be empty")
    collection.delete(ids=[memory_id])
    return {"id": memory_id, "deleted": True}
