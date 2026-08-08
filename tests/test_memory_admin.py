import unittest

from BackEnd.memory_admin import add_memory, delete_memory, list_memories, update_memory


class _Vector:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class FakeEmbedModel:
    def __init__(self):
        self.inputs = []

    def encode(self, value, normalize_embeddings=True):
        self.inputs.append(value)
        return _Vector([[0.3, 0.7] for _ in value])


class FakeCollection:
    def __init__(self):
        self.rows = {
            "mem_1": {
                "document": "主人喜欢安静的夜晚 | 夜晚",
                "metadata": {
                    "source": "dynamic_memory",
                    "timestamp": "2026-08-06 21:00",
                    "status": "approved",
                },
            },
            "knowledge_1": {
                "document": "芯宝是桌面陪伴机器人",
                "metadata": {
                    "source": "knowledge.md",
                    "type": "knowledge",
                },
            },
        }
        self.deleted_ids = []

    def get(self, where=None, include=None):
        rows = self.rows
        if where:
            rows = {
                key: value
                for key, value in rows.items()
                if all(value["metadata"].get(k) == v for k, v in where.items())
            }
        return {
            "ids": list(rows.keys()),
            "documents": [value["document"] for value in rows.values()],
            "metadatas": [value["metadata"] for value in rows.values()],
        }

    def upsert(self, ids, documents, embeddings, metadatas):
        for item_id, document, metadata in zip(ids, documents, metadatas):
            self.rows[item_id] = {
                "document": document,
                "metadata": metadata,
                "embedding": embeddings[0],
            }

    def delete(self, ids):
        self.deleted_ids.extend(ids)
        for item_id in ids:
            self.rows.pop(item_id, None)


class MemoryAdminTest(unittest.TestCase):
    def test_lists_only_dynamic_memories(self):
        memories = list_memories(FakeCollection())

        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["id"], "mem_1")
        self.assertEqual(memories[0]["text"], "主人喜欢安静的夜晚 | 夜晚")
        self.assertEqual(memories[0]["status"], "approved")

    def test_adds_manual_memory_with_embedding_and_metadata(self):
        collection = FakeCollection()
        embed_model = FakeEmbedModel()

        created = add_memory(
            collection,
            embed_model,
            "主人喜欢热牛奶 | 牛奶",
            timestamp="2026-08-06 22:00",
        )

        self.assertIn(created["id"], collection.rows)
        self.assertEqual(collection.rows[created["id"]]["document"], "主人喜欢热牛奶 | 牛奶")
        self.assertEqual(collection.rows[created["id"]]["metadata"]["source"], "dynamic_memory")
        self.assertEqual(collection.rows[created["id"]]["metadata"]["status"], "approved")
        self.assertEqual(embed_model.inputs, [["主人喜欢热牛奶 | 牛奶"]])

    def test_updates_existing_memory_text_and_embedding(self):
        collection = FakeCollection()
        embed_model = FakeEmbedModel()

        updated = update_memory(
            collection,
            embed_model,
            "mem_1",
            "主人喜欢下雨天写代码 | 下雨天,代码",
        )

        self.assertEqual(updated["id"], "mem_1")
        self.assertEqual(collection.rows["mem_1"]["document"], "主人喜欢下雨天写代码 | 下雨天,代码")
        self.assertEqual(collection.rows["mem_1"]["metadata"]["source"], "dynamic_memory")
        self.assertEqual(embed_model.inputs, [["主人喜欢下雨天写代码 | 下雨天,代码"]])

    def test_deletes_memory_by_id(self):
        collection = FakeCollection()

        delete_memory(collection, "mem_1")

        self.assertEqual(collection.deleted_ids, ["mem_1"])
        self.assertNotIn("mem_1", collection.rows)


if __name__ == "__main__":
    unittest.main()
