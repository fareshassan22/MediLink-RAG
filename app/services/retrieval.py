from typing import List, Dict
# …existing code…

def score_answer(answer: str, sources: List[Dict]) -> float:
    # kept for backwards compatibility; delegate to compute_confidence
    return compute_confidence([s.get("score", 0.0) for s in sources])

def compute_confidence(scores: List[float], grounding: float = 1.0) -> float:
    """
    Turn a list of retrieval scores into a 0‑1 value.  Assumes the scores are
    cosine similarities in [0,1] (adjust the normalisation if you use
    a different metric).  Multiplying by `grounding` lets you incorporate the
    model’s self‑reported grounding score.
    """
    if not scores:
        return 0.0
    max_sim = max(scores)
    max_sim = max(0.0, min(1.0, max_sim))
    return float(max_sim * grounding)

# …existing code…