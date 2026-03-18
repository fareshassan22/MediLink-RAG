from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np

from app.indexing.embedder import embed_texts


# ── Sentence splitting (Arabic + English) ──


def _split_sentences(text: str) -> List[str]:
    """Split on period, question mark (Arabic & Latin), exclamation, and newlines."""
    parts = re.split(r"[.؟?!\n]+", text)
    return [s.strip() for s in parts if len(s.strip()) > 5]


# ── Medical-specific hallucination patterns ──

_FABRICATED_PATTERNS = [
    # Invented dosages not in context (e.g. "take 500mg twice daily")
    re.compile(r"\b\d+\s*(?:mg|ml|mcg|iu|units?)\b", re.IGNORECASE),
    # Percentage statistics (e.g. "90% of patients")
    re.compile(r"\b\d{1,3}(?:\.\d+)?%", re.IGNORECASE),
    # Drug brand names pattern (capitalized multi-word)
    re.compile(r"\b[A-Z][a-z]+(?:®|™)\b"),
]


def _extract_medical_claims(text: str) -> List[str]:
    """Extract sentences that contain medical-specific claims (numbers, dosages, stats)."""
    sentences = _split_sentences(text)
    claims = []
    for s in sentences:
        for pat in _FABRICATED_PATTERNS:
            if pat.search(s):
                claims.append(s)
                break
    return claims


# ── Core verification ──


def verify_grounding(
    answer: str, retrieved_texts: List[str], threshold: float = 0.6
) -> Tuple[bool, float]:
    """Verify grounding using semantic similarity between answer sentences and
    retrieved texts.

    Returns (is_grounded, grounding_score) where grounding_score is in [0, 1].
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return False, 0.0

    sent_embs = embed_texts(sentences)
    ctx_embs = embed_texts(retrieved_texts) if retrieved_texts else []

    if len(ctx_embs) == 0:
        return False, 0.0

    sent_embs = np.array(sent_embs)
    ctx_embs = np.array(ctx_embs)

    sims = sent_embs @ ctx_embs.T
    max_sims = sims.max(axis=1)

    grounding_score = float(np.clip(max_sims.mean(), 0.0, 1.0))
    is_grounded = grounding_score >= threshold

    return is_grounded, grounding_score


def verify_claims(
    answer: str, retrieved_texts: List[str], threshold: float = 0.6
) -> Tuple[bool, List[str]]:
    """Return (all_claims_supported, unsupported_claims_list)."""
    sentences = _split_sentences(answer)
    if not sentences:
        return False, sentences

    sent_embs = embed_texts(sentences)
    ctx_embs = embed_texts(retrieved_texts) if retrieved_texts else []

    if len(ctx_embs) == 0:
        return False, sentences

    sent_embs = np.array(sent_embs)
    ctx_embs = np.array(ctx_embs)
    sims = sent_embs @ ctx_embs.T
    max_sims = sims.max(axis=1)

    unsupported = [s for s, sim in zip(sentences, max_sims) if float(sim) < threshold]
    all_ok = len(unsupported) == 0
    return all_ok, unsupported


def check_hallucination(
    answer: str, retrieved_texts: List[str], threshold: float = 0.55
) -> Tuple[bool, float, List[str]]:
    """Full hallucination check combining grounding + claim verification.

    Returns (has_hallucination, risk_score, flagged_sentences).
      - has_hallucination: True if likely fabrication detected
      - risk_score: 0.0 (no risk) → 1.0 (high risk)
      - flagged_sentences: sentences that appear unsupported
    """
    if not answer or not answer.strip():
        return False, 0.0, []

    # 1. Overall grounding score
    _, grounding_score = verify_grounding(answer, retrieved_texts, threshold)

    # 2. Claim-level check
    _, unsupported = verify_claims(answer, retrieved_texts, threshold)

    # 3. Check medical-specific claims (dosages, stats) against context
    medical_claims = _extract_medical_claims(answer)
    context_joined = " ".join(retrieved_texts).lower()
    fabricated_medical = []
    for claim in medical_claims:
        # Extract the specific number/dosage from the claim
        numbers = re.findall(r"\b\d+(?:\.\d+)?", claim)
        # If the claim has numbers not found anywhere in context, flag it
        if numbers and not any(n in context_joined for n in numbers):
            if claim not in unsupported:
                fabricated_medical.append(claim)

    all_flagged = list(set(unsupported + fabricated_medical))
    total_sentences = len(_split_sentences(answer))

    # Risk score: weighted combination
    unsupported_ratio = len(all_flagged) / max(total_sentences, 1)
    risk_score = 0.6 * (1.0 - grounding_score) + 0.4 * unsupported_ratio
    risk_score = float(np.clip(risk_score, 0.0, 1.0))

    has_hallucination = risk_score > 0.5 or len(all_flagged) > total_sentences * 0.5

    return has_hallucination, risk_score, all_flagged
