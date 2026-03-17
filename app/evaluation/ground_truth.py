from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import cfg
from app.indexing.vector_store import VectorStore
from app.indexing.embedder import embed_texts


def _doc_matches_keywords(doc_text: str, keywords: List[str]) -> bool:
    """Check if a document contains at least one expected keyword."""
    doc_lower = doc_text.lower()
    for kw in keywords:
        if kw.lower() in doc_lower:
            return True
    return False


def build_ground_truth_doc_ids(
    examples: List[Dict],
    vs: VectorStore,
    top_k_gt: int = 5,
    score_threshold: float = 0.3,
) -> List[Optional[List[str]]]:
    """Build ground truth doc_id lists using a combination of keyword matching
    and semantic similarity for cross-language support."""
    result = []
    for ex in examples:
        matching_ids = set()

        keywords = ex.get("expected_keywords", [])
        for doc in vs.documents:
            if _doc_matches_keywords(doc.text, keywords):
                matching_ids.add(doc.doc_id)

        if not matching_ids:
            question = ex.get("question") or ex.get("query", "")
            if question:
                q_emb = embed_texts([question])[0]
                hits = vs.search(q_emb, k=top_k_gt)
                for hit in hits:
                    if hit.get("score", 0) >= score_threshold:
                        matching_ids.add(hit["doc_idx"])

        if not matching_ids and keywords:
            for kw in keywords[:3]:
                kw_emb = embed_texts([kw])[0]
                hits = vs.search(kw_emb, k=3)
                for hit in hits:
                    if hit.get("score", 0) >= score_threshold:
                        matching_ids.add(hit["doc_idx"])

        result.append(list(matching_ids) if matching_ids else None)
    return result


def build_ground_truths(examples: List[Dict]) -> List[Optional[List[str]]]:
    """Build ground truth doc_id lists from eval set.

    Uses explicit ground_truth_ids (curated, verified doc_ids).
    Falls back to ground_truth_doc_id for legacy eval files.
    """
    result: List[Optional[List[str]]] = []
    for ex in examples:
        gt_ids = ex.get("ground_truth_ids", [])
        if gt_ids:
            result.append(list(gt_ids))
            continue

        gt_id = ex.get("ground_truth_doc_id")
        if gt_id:
            result.append([gt_id])
            continue

        result.append(None)
    return result


def load_ground_truth(eval_path: Path | None = None) -> List[Dict]:
    """Load evaluation data from JSON file.

    Supports two formats:
    1. Array format: [{"query": "...", "ground_truth_ids": [...]}]
    2. Object format: {"queries": [{"query": "...", "ground_truth_ids": [...]}]}
    """
    eval_path = eval_path or cfg.EVAL_SET_PATH
    import json

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "queries" in data:
        return data["queries"]

    return data


def get_ground_truth_stats(ground_truths: List[Optional[List[str]]]) -> Dict[str, int]:
    """Get statistics about ground truth data."""
    valid_count = sum(1 for gt in ground_truths if gt is not None)
    total_matches = sum(len(gt) for gt in ground_truths if gt is not None)
    return {
        "total_queries": len(ground_truths),
        "valid_queries": valid_count,
        "total_matching_docs": total_matches,
    }
