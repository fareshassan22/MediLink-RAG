import json
import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Optional

try:
    import faiss

    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False


class Document:
    _id_counter = 0

    def __init__(self, text: str, embedding: np.ndarray, metadata: Dict):
        self.doc_id = f"doc_{Document._id_counter}"
        Document._id_counter += 1
        self.text = text
        self.embedding = embedding
        self.metadata = metadata


class VectorStore:
    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.documents: List[Document] = []
        self.embeddings = np.empty((0, dim), dtype="float32")
        self._pending: List[np.ndarray] = []  # buffered embeddings not yet in self.embeddings
        self._faiss_index: Optional[object] = None

    # ---- internal ----
    def _flush_pending(self):
        """Merge buffered embeddings into the main array."""
        if not self._pending:
            return
        stacked = np.vstack(self._pending)
        if len(self.embeddings) == 0:
            self.embeddings = stacked
        else:
            self.embeddings = np.vstack([self.embeddings, stacked])
        self._pending.clear()

    def _rebuild_faiss(self):
        """Build/rebuild the FAISS IndexFlatIP from current embeddings."""
        self._flush_pending()
        if not _HAS_FAISS or len(self.embeddings) == 0:
            self._faiss_index = None
            return
        index = faiss.IndexFlatIP(self.dim)
        # Normalise so IP == cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        index.add((self.embeddings / norms).astype("float32"))
        self._faiss_index = index

    # ---- public API ----
    def add(self, text: str, embedding: np.ndarray, metadata: Dict):
        if embedding is None:
            raise ValueError("Embedding cannot be None")

        embedding = embedding.flatten().astype("float32")
        if len(embedding) != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch. Expected {self.dim}, got {len(embedding)}"
            )

        metadata.setdefault("page", None)
        metadata.setdefault("source", "Medical Textbook")
        metadata.setdefault("title", metadata["source"])

        doc = Document(text, embedding, metadata)
        self.documents.append(doc)
        self._pending.append(embedding.reshape(1, -1))
        # Invalidate FAISS — will be rebuilt lazily on next search
        self._faiss_index = None

    def get_doc(self, idx: int) -> Document:
        if 0 <= idx < len(self.documents):
            return self.documents[idx]
        raise IndexError(f"Document index {idx} out of range")

    def search(self, query_embedding: np.ndarray, k: int = 10, top_k: int = None) -> List[Dict]:
        """Search by embedding similarity.

        Uses FAISS IndexFlatIP when available, falls back to numpy dot product.
        """
        k = top_k if top_k is not None else k
        if len(self.documents) == 0:
            return []

        self._flush_pending()
        query_embedding = query_embedding.flatten().astype("float32")
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        k = min(k, len(self.documents))

        if _HAS_FAISS:
            if self._faiss_index is None:
                self._rebuild_faiss()
            scores_arr, indices_arr = self._faiss_index.search(
                query_norm.reshape(1, -1), k
            )
            top_k_indices = indices_arr[0]
            top_k_scores = scores_arr[0]
        else:
            embeddings_norm = self.embeddings / (
                np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
            )
            scores = np.dot(embeddings_norm, query_norm)
            top_k_indices = np.argsort(scores)[::-1][:k]
            top_k_scores = scores[top_k_indices]

        results = []
        for rank, (idx, score) in enumerate(zip(top_k_indices, top_k_scores)):
            idx = int(idx)
            if idx < 0 or idx >= len(self.documents):
                continue
            doc = self.documents[idx]
            results.append(
                {
                    "doc_idx": idx,
                    "rank": rank,
                    "text": doc.text,
                    "score": float(score),
                    "dense_score": float(score),
                    "metadata": doc.metadata,
                    "page": doc.metadata.get("page"),
                    "source": doc.metadata.get("source"),
                    "title": doc.metadata.get("title"),
                }
            )
        return results

    # Backward-compat alias
    def search_dict(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        return self.search(query_embedding, k=k)

    # ---- persistence ----
    def save(self, path: str):
        self._flush_pending()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "dim": self.dim,
            "documents": [
                {
                    "text": doc.text,
                    "embedding": doc.embedding,
                    "metadata": doc.metadata,
                }
                for doc in self.documents
            ],
        }
        with open(path / "vector_store.pkl", "wb") as f:
            pickle.dump(data, f)
        print(f"Vector store saved: {len(self.documents)} documents")

    def load(self, path: str):
        path = Path(path)
        with open(path / "vector_store.pkl", "rb") as f:
            data = pickle.load(f)

        self.dim = data["dim"]
        self.documents = []
        embeddings_list = []

        for idx, doc_data in enumerate(data["documents"]):
            doc = Document(
                text=doc_data["text"],
                embedding=doc_data["embedding"],
                metadata=doc_data["metadata"],
            )
            doc.doc_id = f"doc_{idx}"
            self.documents.append(doc)
            embeddings_list.append(doc_data["embedding"].reshape(1, -1))

        if embeddings_list:
            self.embeddings = np.vstack(embeddings_list).astype("float32")
        else:
            self.embeddings = np.empty((0, self.dim), dtype="float32")

        # Build FAISS index immediately
        self._rebuild_faiss()
        engine = "FAISS" if self._faiss_index is not None else "numpy"
        print(f"Vector store loaded: {len(self.documents)} documents ({engine})")
