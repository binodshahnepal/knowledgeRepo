import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def embed_texts(self, texts):
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.array(vectors, dtype=np.float32)

    def build(self, chunks_with_metadata):
        if not chunks_with_metadata:
            raise ValueError("No chunks provided.")

        texts = [item["text"] for item in chunks_with_metadata]
        embeddings = self.embed_texts(texts)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        self.documents = chunks_with_metadata

    def search(self, query: str, top_k: int = 5):
        if self.index is None or not self.documents:
            return []

        query_vector = self.embed_texts([query])
        distances, indices = self.index.search(query_vector, min(top_k, len(self.documents)))

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if 0 <= idx < len(self.documents):
                item = dict(self.documents[idx])
                item["distance"] = float(dist)
                results.append(item)
        return results

    def save(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        with open(os.path.join(folder_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)
        with open(os.path.join(folder_path, "meta.pkl"), "wb") as f:
            pickle.dump({"model_name": self.model_name}, f)

    @classmethod
    def load(cls, folder_path: str):
        meta_path = os.path.join(folder_path, "meta.pkl")
        index_path = os.path.join(folder_path, "index.faiss")
        docs_path = os.path.join(folder_path, "documents.pkl")

        if not (os.path.exists(meta_path) and os.path.exists(index_path) and os.path.exists(docs_path)):
            return None

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        store = cls(meta["model_name"])
        store.index = faiss.read_index(index_path)

        with open(docs_path, "rb") as f:
            store.documents = pickle.load(f)

        return store