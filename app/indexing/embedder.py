import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

import torch

# BGE-M3: Multilingual model with excellent Arabic support
# Use GPU 1 if available (GPU 0 reserved for LLM), else CPU
_embed_device = "cuda:1" if torch.cuda.device_count() > 1 else ("cuda:0" if torch.cuda.is_available() else "cpu")
_model = SentenceTransformer("BAAI/bge-m3", device=_embed_device)


class Embedder:
    @staticmethod
    def get():
        return _model


def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """
    Create embeddings using BAAI/bge-m3 (1024-dim).
    BGE-M3 supports 100+ languages including Arabic.
    Embeddings are normalized for cosine similarity.
    """

    if not texts:
        return []

    embeddings = _model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
    )

    if len(texts) == 1:
        embeddings = embeddings.reshape(1, -1)

    return [emb.astype("float32") for emb in embeddings]
