"""Reranker module.

The old amberoad/bert-multilingual-passage-reranking-msmarco model actively
degraded medical Arabic retrieval (Recall@1=0.004).  The module now defaults
to NO reranking unless a model name is explicitly configured in config.
"""

from typing import List, Dict, Optional
import numpy as np

_cross_encoder = None  # lazy-loaded


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    from app.core.config import cfg

    model_name = cfg.RERANKER_MODEL_NAME
    if not model_name:
        return None

    import os
    import torch
    from sentence_transformers import CrossEncoder

    # Use GPU 1 if available, to separate from LLM on GPU 0
    device = "cuda:1" if torch.cuda.device_count() > 1 else None
    _cross_encoder = CrossEncoder(model_name, device=device)
    return _cross_encoder


class DefaultReranker:
    """Reranker that delegates to a CrossEncoder if one is configured."""

    def __init__(self):
        self.cross_encoder = _get_cross_encoder()

    def rerank(
        self, query: str, candidates: List[tuple], top_k: int = 10
    ) -> List[float]:
        if not candidates or self.cross_encoder is None:
            return [0.0] * len(candidates)

        pairs = [[query, text] for _, text in candidates]
        scores = self.cross_encoder.predict(pairs)
        if len(scores.shape) > 1 and scores.shape[1] > 1:
            scores = scores[:, 0]
        return scores.tolist()


default_reranker = DefaultReranker()


def rerank(query: str, documents: List[Dict], top_k: int = 10) -> List[Dict]:
    """Rerank documents using cross-encoder.  Returns input unchanged when
    no reranker model is configured."""
    encoder = _get_cross_encoder()
    if not documents or encoder is None:
        return documents[:top_k]

    # Keep all candidates for reranking (don't pre-filter)
    pairs = [[query, doc.get("text", "")] for doc in documents]
    scores = encoder.predict(pairs)
    if len(scores.shape) > 1 and scores.shape[1] > 1:
        scores = scores[:, 0]

    mn, mx = float(np.min(scores)), float(np.max(scores))
    if mx > mn:
        scores_norm = (scores - mn) / (mx - mn)
    else:
        scores_norm = np.ones_like(scores) * 0.5

    for doc, raw, norm in zip(documents, scores, scores_norm):
        doc["rerank_score"] = float(raw)
        doc["rerank_score_normalized"] = float(norm)

    reranked = sorted(
        documents, key=lambda x: x.get("rerank_score_normalized", 0.0), reverse=True
    )
    return reranked[:top_k]
