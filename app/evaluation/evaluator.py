from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.utils.seed import set_seed, DEFAULT_SEED

set_seed(DEFAULT_SEED)

from app.core.config import cfg
from app.indexing.vector_store import VectorStore
from app.indexing.index_pipeline import load_bm25
from app.retrieval.fusion import fuse_scores
from app.retrieval.reranker import default_reranker
from app.safety.grounding import grounding_verification
from app.evaluation.ground_truth import (
    build_ground_truths,
    load_ground_truth,
    get_ground_truth_stats,
)
from app.evaluation.metrics import (
    recall_at_k,
    mrr,
    expected_calibration_error,
    pearson_corr,
    precision_at_k,
    ndcg_at_k,
)

logger = logging.getLogger(__name__)


def _compute_retrieval_metrics(
    ranked_lists: List[List[str]],
    ground_truths: List[Optional[List[str]]],
) -> Dict[str, float]:
    """Compute all retrieval metrics from ranked lists and ground truths."""
    metrics = {}
    for k in [1, 3, 5, 10]:
        metrics[f"recall@{k}"] = recall_at_k(ranked_lists, ground_truths, k)
        metrics[f"precision@{k}"] = precision_at_k(ranked_lists, ground_truths, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_lists, ground_truths, k)
    metrics["mrr"] = mrr(ranked_lists, ground_truths)
    return metrics


class Evaluator:
    def __init__(self, vs: VectorStore, bm25=None) -> None:
        self.vs = vs
        self.bm25 = bm25 or load_bm25()
        self._doc_cache: Dict[str, str] = {}

    def _get_doc_text(self, doc_id: str) -> str:
        """Get document text with caching to avoid repeated lookups."""
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]
        for doc in self.vs.documents:
            if doc.doc_id == doc_id:
                text = doc.text
                self._doc_cache[doc_id] = text
                return text
        return ""

    def _initial_retrieval(
        self, query: str, top_k: int
    ) -> Tuple[List[int], List[float], List[int], List[float]]:
        from app.indexing.embedder import embed_texts

        query_emb = embed_texts([query])[0]
        dense_results = self.vs.search(query_emb, k=top_k)
        dense_idx = [r.get("doc_idx", i) for i, r in enumerate(dense_results)]
        dense_scores = [r.get("score", 0.0) for r in dense_results]

        bm25_idx = []
        bm25_scores = []
        if self.bm25:
            all_scores = self.bm25.get_scores(query)
            sorted_idx = sorted(range(len(all_scores)), key=lambda i: all_scores[i], reverse=True)[:top_k]
            bm25_idx = sorted_idx
            bm25_scores = [all_scores[i] for i in sorted_idx]

        return dense_idx, dense_scores, bm25_idx, bm25_scores

    def rank(
        self, query: str, mode: str = "hybrid", top_k: int = 20
    ) -> Tuple[List[str], Dict[str, float], Dict[int, float]]:
        initial_k = max(cfg.TOP_K_INITIAL, top_k)
        dens_idx, dens_scores, _, bm25_scores_initial = self._initial_retrieval(
            query, top_k=initial_k
        )

        bm25_all_scores_raw = self.bm25.get_scores(query) if self.bm25 else []
        bm25_idx = sorted(range(len(bm25_all_scores_raw)), key=lambda i: bm25_all_scores_raw[i], reverse=True)[:initial_k] if bm25_all_scores_raw else []
        candidate_idx = list(dict.fromkeys(dens_idx + bm25_idx))

        bm25_all_scores = bm25_all_scores_raw if bm25_all_scores_raw else [0.0] * len(self.vs.documents)

        dense_score_map = {i: 0.0 for i in candidate_idx}
        for i, s in zip(dens_idx, dens_scores):
            dense_score_map[i] = s

        dense_aligned = [dense_score_map[i] for i in candidate_idx]
        bm25_aligned = [bm25_all_scores[i] for i in candidate_idx]

        if mode == "dense_only":
            fused = dense_aligned
        elif mode == "bm25_only":
            fused = bm25_aligned
        else:
            fused = fuse_scores(dense_aligned, bm25_aligned)

        ranked = sorted(
            list(zip(candidate_idx, fused)), key=lambda x: (x[1], -x[0]), reverse=True
        )[:top_k]

        rerank_map = {}
        if mode == "hybrid_plus_rerank":
            # Rerank only the top 10 for a higher-quality candidate set
            rerank_top = ranked[:10]
            candidates = [(str(i), self.vs.get_doc(i).text) for i, _ in rerank_top]
            # Translate query for the English-only cross-encoder reranker
            from app.retrieval.query_translator import translate_query

            rerank_query = translate_query(query)
            rerank_scores = default_reranker.rerank(rerank_query, candidates)

            reranked_with_scores = list(zip(rerank_scores, rerank_top))
            reranked_with_scores.sort(key=lambda x: (x[0], -int(x[1][0])), reverse=True)
            # Combine: reranked top + remaining non-reranked
            reranked_ids = {doc_idx for _, (doc_idx, _) in reranked_with_scores}
            remaining = [(i, s) for i, s in ranked if i not in reranked_ids]
            ranked = [score_doc[1] for score_doc in reranked_with_scores] + remaining

            for idx, (score, (doc_idx, _)) in enumerate(reranked_with_scores):
                rerank_map[int(doc_idx)] = score

        doc_ids = [self.vs.get_doc(i).doc_id for i, _ in ranked]

        # Top-document scores reflect the actual mode ranking
        top_idx = ranked[0][0] if ranked else -1
        top_dense = dense_score_map.get(top_idx, 0.0) if top_idx >= 0 else 0.0
        top_bm25 = float(bm25_all_scores[top_idx]) if top_idx >= 0 else 0.0

        diagnostics = {
            "mean_dense_score": float(np.mean(dense_aligned)) if dense_aligned else 0.0,
            "mean_bm25_score": float(np.mean(bm25_aligned)) if bm25_aligned else 0.0,
            "top_dense_score": top_dense,
            "top_bm25_score": top_bm25,
            "num_candidates": len(candidate_idx),
        }

        if rerank_map:
            diagnostics["mean_rerank_score"] = float(np.mean(list(rerank_map.values())))

        return doc_ids, diagnostics, rerank_map

    def evaluate(self, eval_path: Path | None = None) -> None:
        eval_path = eval_path or cfg.EVAL_SET_PATH
        logger.info(f"Loading evaluation set from: {eval_path}")

        examples = load_ground_truth(eval_path)

        if not examples:
            logger.warning("Empty evaluation set")
            return

        ground_truths = build_ground_truths(examples)
        stats = get_ground_truth_stats(ground_truths)
        print(
            f"Ground truth: {stats['valid_queries']}/{stats['total_queries']} queries matched, "
            f"{stats['total_matching_docs']} total matching docs"
        )
        logger.info(
            f"Evaluating {stats['total_queries']} queries "
            f"({stats['valid_queries']} with matches, {stats['total_matching_docs']} total matching docs)"
        )

        modes = ["dense_only", "bm25_only", "hybrid", "hybrid_plus_rerank"]
        results = []

        for mode in modes:
            print(f"Evaluating mode: {mode}...")
            logger.info(f"Evaluating mode: {mode}")
            ranked_lists = []
            mean_grounding = []
            confidences = []
            rerank_scores_list = []
            labels = []

            for ex, gt in zip(examples, ground_truths):
                q = ex.get("question", "") or ex.get("query", "")

                doc_ids, diag, rerank_map = self.rank(q, mode=mode, top_k=20)
                ranked_lists.append(doc_ids)

                top_doc = doc_ids[0] if doc_ids else ""
                top_text = self._get_doc_text(top_doc) if top_doc else ""

                # Grounding: use provided answer if available, else keyword coverage
                # with semantic fallback for cross-lingual scenarios
                answer = ex.get("answer", "")
                if answer and top_text:
                    grounding_score, _ = grounding_verification(answer, top_text)
                elif top_text:
                    keywords = ex.get("expected_keywords", [])
                    if keywords:
                        # Try keyword matching first
                        found = sum(
                            1 for kw in keywords if kw.lower() in top_text.lower()
                        )
                        kw_score = found / len(keywords)
                        if kw_score > 0:
                            grounding_score = kw_score
                        else:
                            # Semantic fallback: use expected keywords as proxy answer
                            grounding_score, _ = grounding_verification(
                                " ".join(keywords), top_text
                            )
                    else:
                        grounding_score = 0.0
                else:
                    grounding_score = 0.0
                mean_grounding.append(grounding_score)

                if mode == "hybrid_plus_rerank" and rerank_map:
                    top_rerank = rerank_map.get(int(self._find_doc_idx(top_doc)), 0.0)
                else:
                    top_rerank = 0.0

                rerank_scores_list.append(top_rerank)

                # Use mode-specific top-document score for confidence
                confidence = self._compute_confidence(
                    diag.get("top_dense_score", 0.0),
                    grounding_score,
                    top_rerank,
                    diag.get("top_bm25_score", 0.0),
                    mode=mode,
                )
                confidences.append(confidence)

                # Check if top doc is in ground truth
                is_hit = 1 if gt and top_doc in gt else 0
                labels.append(is_hit)

            retrieval_metrics = _compute_retrieval_metrics(ranked_lists, ground_truths)

            mean_grounding_val = (
                float(np.mean(mean_grounding)) if mean_grounding else 0.0
            )
            mean_conf = float(np.mean(confidences)) if confidences else 0.0

            labels_arr = np.array(labels, dtype=float)
            confidences_arr = np.array(confidences)

            pearson = pearson_corr(confidences_arr, labels_arr)
            ece = expected_calibration_error(confidences_arr, labels_arr, n_bins=5)

            result = {
                "mode": mode,
                "recall@1": retrieval_metrics.get("recall@1", 0.0),
                "recall@5": retrieval_metrics.get("recall@5", 0.0),
                "recall@10": retrieval_metrics.get("recall@10", 0.0),
                "mrr": retrieval_metrics.get("mrr", 0.0),
                "mean_grounding": mean_grounding_val,
                "mean_confidence": mean_conf,
                "pearson": pearson,
                "ece": ece,
            }
            results.append(result)
            print(
                f"  {mode}: Recall@1={result['recall@1']:.3f}, Recall@5={result['recall@5']:.3f}, "
                f"Recall@10={result['recall@10']:.3f}, MRR={result['mrr']:.3f}, "
                f"Grounding={result['mean_grounding']:.3f}"
            )
            logger.info(
                f"  {mode}: Recall@5={result['recall@5']:.3f}, MRR={result['mrr']:.3f}, "
                f"Grounding={result['mean_grounding']:.3f}"
            )

        cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        csv_path = cfg.RESULTS_DIR / "retrieval_metrics.csv"

        with open(csv_path, "w", encoding="utf-8", newline="") as csvfile:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

        logger.info(f"Results written to: {csv_path}")

    def _find_doc_idx(self, doc_id: str) -> int:
        """Find document index by doc_id."""
        for idx, doc in enumerate(self.vs.documents):
            if doc.doc_id == doc_id:
                return idx
        return -1

    def _compute_confidence(
        self,
        retrieval_score: float,
        grounding_score: float,
        rerank_score: float,
        bm25_score: float = 0.0,
        mode: str = "dense_only",
    ) -> float:
        """Compute combined confidence from available signals per mode.

        Weights are redistributed based on which signals are active so that
        the full [0, 1] range is reachable for every mode.
        """
        retrieval_norm = max(0.0, min(1.0, (retrieval_score + 1.0) / 2.0))
        grounding_norm = max(0.0, min(1.0, grounding_score))
        rerank_norm = max(0.0, min(1.0, rerank_score))
        bm25_norm = max(0.0, min(1.0, bm25_score / 30.0)) if bm25_score > 0 else 0.0

        if mode == "hybrid_plus_rerank":
            confidence = (
                0.30 * grounding_norm
                + 0.25 * retrieval_norm
                + 0.25 * rerank_norm
                + 0.20 * bm25_norm
            )
        elif mode == "hybrid":
            confidence = (
                0.40 * grounding_norm + 0.35 * retrieval_norm + 0.25 * bm25_norm
            )
        elif mode == "bm25_only":
            confidence = 0.50 * grounding_norm + 0.50 * bm25_norm
        else:  # dense_only
            confidence = 0.45 * grounding_norm + 0.55 * retrieval_norm

        return float(max(0.0, min(1.0, confidence)))
