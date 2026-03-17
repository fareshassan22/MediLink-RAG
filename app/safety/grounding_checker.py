import numpy as np
from app.indexing.embedder import embed_texts


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_grounding_score(answer: str, context_chunks: list) -> float:
    """
    Computes grounding score between generated answer
    and all retrieved context chunks.
    Returns score between 0 and 1.
    """

    if not context_chunks:
        return 0.0

    texts = [answer] + context_chunks
    embeddings = embed_texts(texts)

    answer_emb = embeddings[0]
    context_embs = embeddings[1:]

    similarities = [
        cosine_similarity(answer_emb, chunk_emb)
        for chunk_emb in context_embs
    ]

    max_similarity = max(similarities)

    # Use raw cosine similarity — already [0,1] for normalized embeddings
    # Clamp to [0,1] as a safety net
    return float(max(0.0, min(1.0, max_similarity)))


def verify_grounding(answer, context_chunks, threshold=0.65):
    score = compute_grounding_score(answer, context_chunks)

    if score < threshold:
        return False, score

    return True, score