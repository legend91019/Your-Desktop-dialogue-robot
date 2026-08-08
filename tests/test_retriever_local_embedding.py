import importlib.util
import contextlib
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path


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
        if isinstance(value, list):
            return _Vector([[0.1, 0.2] for _ in value])
        return _Vector([0.1, 0.2])


class FakeCollection:
    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.embeddings = []

    def get(self, include=None):
        return {"ids": []}

    def upsert(self, ids, documents, metadatas, embeddings):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.embeddings.extend(embeddings)

    def query(self, query_embeddings, n_results):
        return {
            "documents": [self.documents[:n_results]],
            "metadatas": [self.metadatas[:n_results]],
        }


class FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_or_create_collection(self, name):
        return self.collection


class RetrieverLocalEmbeddingTest(unittest.TestCase):
    def test_uses_injected_local_embedding_model(self):
        collection = FakeCollection()

        fake_chromadb = types.ModuleType("chromadb")
        fake_chromadb.PersistentClient = lambda path: FakeClient(collection)
        sys.modules["chromadb"] = fake_chromadb

        fake_sentence_transformers = types.ModuleType("sentence_transformers")

        def fail_if_remote_model_is_loaded(*args, **kwargs):
            raise AssertionError("remote SentenceTransformer should not be instantiated")

        fake_sentence_transformers.SentenceTransformer = fail_if_remote_model_is_loaded
        sys.modules["sentence_transformers"] = fake_sentence_transformers

        module_path = Path(__file__).resolve().parents[1] / "utils" / "Retriever" / "retriever.py"
        spec = importlib.util.spec_from_file_location("retriever_under_test", module_path)
        retriever_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(retriever_module)

        with tempfile.TemporaryDirectory() as tmp_dir:
            md_path = Path(tmp_dir) / "knowledge.md"
            md_path.write_text("# title\nhello world", encoding="utf-8")

            embed_model = FakeEmbedModel()
            with contextlib.redirect_stdout(io.StringIO()):
                retrieve = retriever_module.create_rag_retriever(
                    str(md_path),
                    embed_model=embed_model,
                    collection=collection,
                    top_k=1,
                )

            result = retrieve("hello?")

        self.assertIn("hello world", result)
        self.assertIn(["hello world"], embed_model.inputs)
        self.assertIn("hello?", embed_model.inputs)


if __name__ == "__main__":
    unittest.main()
