"""Weighted hybrid fusion for combining dense and BM25 retrieval results.

Replaces naive RRF with score-normalized weighted fusion.
Dense retrieval is primary (0.8), BM25 is secondary (0.2).
Includes intent-aware weight adjustment and post-fusion filtering.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Dict, List, Tuple

from app.core.config import cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent detection
# ---------------------------------------------------------------------------

_INTENT_PATTERNS: Dict[str, List[str]] = {
    "symptoms": [
        r"أعراض", r"عوارض", r"علامات", r"يشعر", r"يعاني",
        r"symptoms", r"signs", r"feel", r"experiencing",
    ],
    "causes": [
        r"أسباب", r"سبب", r"لماذا", r"يسبب", r"عوامل",
        r"causes", r"why", r"etiology", r"risk.?factors",
    ],
    "treatment": [
        r"علاج", r"دواء", r"أدوية", r"كيف.?يعالج", r"عملية", r"جراحة",
        r"treatment", r"therapy", r"medication", r"cure", r"manage",
    ],
    "diagnosis": [
        r"تشخيص", r"فحص", r"تحليل", r"كشف",
        r"diagnosis", r"test", r"detect", r"screen",
    ],
    "prevention": [
        r"وقاية", r"حماية", r"تجنب", r"منع",
        r"prevention", r"prevent", r"avoid", r"protect",
    ],
}

# Intent -> (dense_weight, bm25_weight)
_INTENT_WEIGHTS: Dict[str, Tuple[float, float]] = {
    "symptoms":   (0.75, 0.25),   # BM25 slightly higher: keyword matching helps
    "causes":     (0.80, 0.20),   # balanced
    "treatment":  (0.85, 0.15),   # dense higher: semantic understanding matters
    "diagnosis":  (0.80, 0.20),
    "prevention": (0.80, 0.20),
    "general":    (0.80, 0.20),
}


def detect_intent(query: str) -> str:
    """Classify query intent from surface patterns."""
    query_lower = query.lower()
    best_intent = "general"
    best_count = 0

    for intent, patterns in _INTENT_PATTERNS.items():
        count = sum(1 for p in patterns if re.search(p, query_lower))
        if count > best_count:
            best_count = count
            best_intent = intent

    return best_intent


def get_intent_weights(intent: str) -> Tuple[float, float]:
    """Return (dense_weight, bm25_weight) for the detected intent."""
    return _INTENT_WEIGHTS.get(intent, (cfg.WEIGHT_DENSE, cfg.WEIGHT_BM25))


# ---------------------------------------------------------------------------
# Score normalization
# ---------------------------------------------------------------------------

def normalize_scores(scores: List[float]) -> List[float]:
    """Min-max normalize scores to [0, 1]."""
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx - mn < 1e-9:
        return [1.0] * len(scores) if mx > 0 else [0.0] * len(scores)
    return [(s - mn) / (mx - mn) for s in scores]


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _get_text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:16]


def deduplicate_results(
    results: List[Dict], similarity_threshold: float = 0.85
) -> List[Dict]:
    """Remove duplicate or highly similar documents using Jaccard overlap."""
    if not results:
        return []

    seen_hashes: set = set()
    seen_texts: list = []
    deduplicated: list = []

    for doc in results:
        text = doc.get("text", "")
        if not text:
            continue

        text_hash = _get_text_hash(text)
        if text_hash in seen_hashes:
            continue

        is_duplicate = False
        text_words = set(text.lower().split())

        for seen_text in seen_texts:
            seen_words = set(seen_text.lower().split())
            if not seen_words:
                continue
            intersection = len(text_words & seen_words)
            union = len(text_words | seen_words)
            if union > 0 and intersection / union >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            seen_hashes.add(text_hash)
            seen_texts.append(text)
            deduplicated.append(doc)

    if len(deduplicated) < len(results):
        logger.info(f"Deduplication: {len(results)} -> {len(deduplicated)} documents")

    return deduplicated


# ---------------------------------------------------------------------------
# Weighted fusion (replaces naive RRF)
# ---------------------------------------------------------------------------

def weighted_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    dense_weight: float = 0.8,
    bm25_weight: float = 0.2,
) -> List[Dict]:
    """Combine dense and BM25 results using score-normalized weighted fusion.

    Steps:
      1. Normalize each result set's scores to [0, 1]
      2. Accumulate weighted scores per document (keyed by text)
      3. Sort by fused score descending

    Returns list of dicts with 'score', 'dense_score', 'bm25_score', 'fusion_score'.
    """
    doc_scores: Dict[str, float] = {}
    doc_data: Dict[str, Dict] = {}

    # --- Normalize dense scores ---
    raw_dense = [d.get("score", 0.0) for d in dense_results]
    norm_dense = normalize_scores(raw_dense)

    for doc, normed in zip(dense_results, norm_dense):
        text = doc.get("text", "")
        if not text:
            continue
        if text not in doc_data:
            doc_data[text] = doc.copy()
            doc_data[text]["dense_score"] = doc.get("score", 0.0)
            doc_data[text]["bm25_score"] = 0.0
            doc_scores[text] = 0.0
        doc_scores[text] += dense_weight * normed

    # --- Normalize BM25 scores ---
    raw_bm25 = [d.get("score", d.get("bm25_score", 0.0)) for d in bm25_results]
    norm_bm25 = normalize_scores(raw_bm25)

    for doc, normed in zip(bm25_results, norm_bm25):
        text = doc.get("text", "")
        if not text:
            continue
        if text not in doc_data:
            doc_data[text] = doc.copy()
            doc_data[text]["dense_score"] = 0.0
            doc_data[text]["bm25_score"] = doc.get("bm25_score", doc.get("score", 0.0))
            doc_scores[text] = 0.0
        else:
            doc_data[text]["bm25_score"] = doc.get("bm25_score", doc.get("score", 0.0))
        doc_scores[text] += bm25_weight * normed

    # --- Agreement boosting: reward docs found by BOTH retrievers ---
    both_retrievers: set = set()
    dense_texts = {d.get("text", "") for d in dense_results if d.get("text")}
    bm25_texts = {d.get("text", "") for d in bm25_results if d.get("text")}
    both_retrievers = dense_texts & bm25_texts

    # --- Build fused ranked list ---
    fused: List[Dict] = []
    for text, fused_score in doc_scores.items():
        entry = doc_data[text].copy()
        if text in both_retrievers:
            fused_score *= 1.2  # 20% bonus for dual-retrieval agreement
            entry["agreement"] = True
        else:
            entry["agreement"] = False
        entry["score"] = fused_score
        entry["fusion_score"] = fused_score
        fused.append(entry)

    fused.sort(key=lambda x: x["score"], reverse=True)

    logger.info(
        f"Weighted fusion: {len(dense_results)} dense + {len(bm25_results)} BM25 "
        f"-> {len(fused)} fused (w_d={dense_weight}, w_b={bm25_weight})"
    )
    return fused


# ---------------------------------------------------------------------------
# Post-fusion filtering — remove low-relevance chunks
# ---------------------------------------------------------------------------

def post_fusion_filter(
    results: List[Dict],
    query: str,
    min_keyword_overlap: float = 0.1,
    top_k: int = 10,
) -> List[Dict]:
    """Filter fused results by keyword overlap with the query.

    Keeps chunks where at least `min_keyword_overlap` fraction of query
    keywords appear in the chunk text. Always keeps the top result.
    For Arabic queries, also checks English translations of query terms.
    """
    if not results:
        return []

    from app.indexing.preprocessing import preprocess_query as _prep

    query_clean = _prep(query).lower()
    query_words = set(query_clean.split())
    # Also add unstemmed words for broader matching
    query_words.update(query.lower().split())

    # For Arabic queries, add English medical term translations so
    # keyword overlap works against the English-only corpus.
    try:
        from app.retrieval.query_expansion import _expand_arabic_medical_terms
        english_terms = _expand_arabic_medical_terms(query)
        for term in english_terms:
            query_words.update(term.lower().split())
    except ImportError:
        pass

    if not query_words:
        return results[:top_k]

    filtered: List[Dict] = []
    for i, doc in enumerate(results):
        text = doc.get("text", "").lower()
        text_words = set(text.split())

        overlap = len(query_words & text_words)
        ratio = overlap / len(query_words)

        # Always keep: (a) top result, (b) docs with sufficient overlap,
        # (c) docs with high dense score
        if i == 0 or ratio >= min_keyword_overlap or doc.get("dense_score", 0) > 0.5:
            filtered.append(doc)

    logger.info(f"Post-fusion filter: {len(results)} -> {len(filtered)} (query words: {len(query_words)})")
    return filtered[:top_k]


# ---------------------------------------------------------------------------
# Main hybrid pipeline entry point
# ---------------------------------------------------------------------------

def hybrid_retrieval_fusion(
    dense_results: List[Dict],
    bm25_results: List[Dict],
    query: str,
    top_k: int = 10,
) -> List[Dict]:
    """Full hybrid retrieval pipeline:
      1. Detect query intent
      2. Weighted fusion with intent-aware weights
      3. Deduplication
      4. Post-fusion keyword filtering
      5. Return top-k

    For Arabic queries against an English corpus, BM25 weight is reduced
    since keyword matching rarely works cross-lingually.
    """
    intent = detect_intent(query)
    w_dense, w_bm25 = get_intent_weights(intent)

    # Detect Arabic and down-weight BM25 (keyword match is unreliable
    # cross-lingually even after translation)
    _has_arabic = any("\u0600" <= c <= "\u06FF" for c in query)
    if _has_arabic:
        w_dense = min(w_dense + 0.10, 0.95)
        w_bm25 = 1.0 - w_dense

    logger.info(f"Intent: {intent} -> weights(dense={w_dense:.2f}, bm25={w_bm25:.2f}, arabic={_has_arabic})")

    # Weighted fusion with normalized scores
    fused = weighted_fusion(dense_results, bm25_results, w_dense, w_bm25)

    # Deduplicate
    fused = deduplicate_results(fused)

    # Post-fusion relevance filter
    fused = post_fusion_filter(fused, query, min_keyword_overlap=cfg.MIN_KEYWORD_OVERLAP, top_k=top_k)

    return fused
