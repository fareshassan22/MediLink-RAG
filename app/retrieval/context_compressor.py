from typing import List, Dict
from app.indexing.embedder import embed_texts
import numpy as np


def _embed_sentences(sentences: List[str]):
    if not sentences:
        return np.array([])
    embs = embed_texts(sentences)
    return np.array(embs)


def compress_context(query: str, sentences: List[Dict], max_tokens: int = 1000) -> List[str]:
    """Compress sentence-level context for LLM prompt.

    - removes duplicate sentences
    - ranks sentences by semantic similarity to the query
    - keeps sentences until approximate token budget (words) is reached
    """
    if not sentences:
        return []

    unique_texts = []
    seen = set()
    for s in sentences:
        text = s.get("sentence") or s.get("text")
        if not text:
            continue
        t = text.strip()
        if t in seen:
            continue
        seen.add(t)
        unique_texts.append(t)

    if not unique_texts:
        return []

    # embed
    query_emb = _embed_sentences([query])
    sent_embs = _embed_sentences(unique_texts)

    if sent_embs.size == 0 or query_emb.size == 0:
        return unique_texts[:10]

    sims = (sent_embs @ query_emb[0])
    ranked_idx = np.argsort(sims)[::-1]

    selected = []
    token_count = 0
    for idx in ranked_idx:
        s = unique_texts[int(idx)]
        tokens = len(s.split())
        # If this sentence exceeds budget but we have nothing yet, truncate and add it
        if token_count + tokens > max_tokens:
            if not selected:
                words = s.split()[:max_tokens]
                selected.append(" ".join(words))
            break
        selected.append(s)
        token_count += tokens

    return selected
