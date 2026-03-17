from typing import List, Tuple
from app.indexing.embedder import embed_texts
import numpy as np


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in text.replace("؟", ".").split(".") if len(s.strip()) > 5]


def verify_grounding(answer: str, retrieved_texts: List[str], threshold: float = 0.6) -> Tuple[bool, float]:
    """Verify grounding using semantic similarity between answer sentences and retrieved texts.

    Returns (is_grounded, grounding_score) where grounding_score is in [0,1].
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return False, 0.0

    # embed sentences and retrieved texts
    sent_embs = embed_texts(sentences)
    ctx_embs = embed_texts(retrieved_texts) if retrieved_texts else []

    if len(ctx_embs) == 0:
        return False, 0.0

    sent_embs = np.array(sent_embs)
    ctx_embs = np.array(ctx_embs)

    # compute max similarity of each sentence to any context chunk
    sims = sent_embs @ ctx_embs.T
    max_sims = sims.max(axis=1)

    # grounding score = mean of max_sims clipped to [0,1]
    grounding_score = float(np.clip(max_sims.mean(), 0.0, 1.0))

    is_grounded = grounding_score >= threshold

    return is_grounded, grounding_score


def verify_claims(answer: str, retrieved_texts: List[str], threshold: float = 0.6) -> Tuple[bool, List[str]]:
    """Return (all_claims_supported, unsupported_claims_list)"""
    sentences = _split_sentences(answer)
    if not sentences:
        return False, sentences

    sent_embs = embed_texts(sentences)
    ctx_embs = embed_texts(retrieved_texts) if retrieved_texts else []

    if len(ctx_embs) == 0:
        return False, sentences

    import numpy as np

    sent_embs = np.array(sent_embs)
    ctx_embs = np.array(ctx_embs)
    sims = sent_embs @ ctx_embs.T
    max_sims = sims.max(axis=1)

    unsupported = [s for s, sim in zip(sentences, max_sims) if float(sim) < threshold]
    all_ok = len(unsupported) == 0
    return all_ok, unsupported
