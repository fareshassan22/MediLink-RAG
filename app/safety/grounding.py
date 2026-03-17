"""Grounding verification module."""

from typing import List, Tuple
import numpy as np
from app.indexing.embedder import embed_texts


def grounding_verification(
    answer: str, context: str, threshold: float = 0.5
) -> Tuple[float, bool]:
    """Verify if answer is grounded in the retrieved context.

    Args:
        answer: The generated answer
        context: The retrieved context
        threshold: Minimum similarity threshold

    Returns:
        (grounding_score, is_grounded) tuple
    """
    answer_sentences = [
        s.strip() for s in answer.replace("؟", ".").split(".") if len(s.strip()) > 10
    ]

    if not answer_sentences:
        return 0.0, False

    # Embed answer sentences and context
    answer_embs = embed_texts(answer_sentences)
    context_embs = embed_texts([context])[0]

    if len(answer_embs) == 0 or context_embs is None:
        return 0.0, False

    # Stack embeddings for batch computation
    answer_embs_matrix = np.vstack(answer_embs)

    # Compute similarity
    sims = np.dot(answer_embs_matrix, context_embs)
    max_sim = float(np.clip(np.max(sims), 0.0, 1.0))

    grounding_score = max_sim
    is_grounded = grounding_score >= threshold

    return grounding_score, is_grounded
