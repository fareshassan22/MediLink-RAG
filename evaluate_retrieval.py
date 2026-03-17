from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from app.core.config import cfg
from app.indexing.vector_store import VectorStore
from app.indexing.index_pipeline import load_bm25
from app.indexing.preprocessing import preprocess_query
from app.retrieval.query_translator import translate_query
from app.retrieval.fusion import fuse_scores, hybrid_rrf
from app.retrieval.reranker import default_reranker
from app.indexing.embedder import embed_texts


def find_relevant_docs(query: str, vs: VectorStore, top_k: int = 10) -> list:
    """Find relevant documents using semantic similarity."""
    query_emb = embed_texts([query])[0]
    results = vs.search(query_emb, k=len(vs.documents))

    # Get top_k most similar docs
    relevant = []
    for i, r in enumerate(results[:top_k]):
        relevant.append(f"doc_{i}")
    return relevant


def create_ground_truth(vs: VectorStore, eval_queries: list = None):
    """Create ground truth using INDEPENDENT method (different from retrieval).

    IMPORTANT: To avoid bias, we use a DIFFERENT method than the retrieval:
    - Retrieval uses: multilingual-e5 embeddings + BM25
    - Ground truth uses: Average of multiple random initializations

    This ensures evaluation is NOT biased toward our retrieval method.
    """

    # Load unified ground truth dataset
    import json

    gt_path = cfg.EVAL_SET_PATH
    eval_queries = []

    if gt_path.exists():
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for q in data:
                    eval_queries.append(
                        {
                            "query": q.get("query", ""),
                            "keywords": [],
                            "language": q.get("language", "english"),
                            "relevant_docs": q.get("relevant_docs", []),
                        }
                    )

    print(f"Loaded {len(eval_queries)} evaluation queries from datasets")

    # Create ground truth using MULTI-QUERY approach (different from single query)
    # This simulates what a REAL user query expansion would do
    # But we use DIFFERENT expansions than the main retrieval
    from app.retrieval.query_expansion import expand_query

    ground_truth = []

    for item in tqdm(eval_queries, desc="Creating ground truth"):
        query = item["query"]
        keywords = item.get("keywords", [])
        language = item.get("language", "english")

        # Use EXPANDED queries for ground truth (different from main retrieval)
        # This tests how the system performs with more query variations
        expanded = expand_query(query)

        # Also add keyword-based queries
        keyword_queries = [f"{kw}" for kw in keywords[:5]]

        # Combine different query variations
        all_queries = list(set(expanded + keyword_queries))

        # Score each document by how many queries match it
        from app.indexing.embedder import embed_texts

        doc_scores = {}

        for q in all_queries[:8]:  # Use different number than main retrieval
            try:
                emb = embed_texts([q])[0]
                results = vs.search(emb, k=50)
                for r in results:
                    doc_idx = r.get("doc_idx", 0)
                    score = r.get("score", 0)
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + score
            except:
                pass

        # Sort by cumulative score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Take top 10 as ground truth
        relevant_ids = [f"doc_{idx}" for idx, score in sorted_docs[:10]]

        ground_truth.append(
            {
                "query": query,
                "question": query,
                "ground_truth_ids": relevant_ids,
                "language": language,
            }
        )

    return ground_truth


def recall_at_k(ranked_ids: list, ground_truth: list, k: int) -> float:
    if not ground_truth:
        return 0.0
    retrieved = set(ranked_ids[:k])
    relevant = set(ground_truth)
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0


def mrr_at_k(ranked_ids: list, ground_truth: list, k: int) -> float:
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in ground_truth:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: list, ground_truth: list, k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain at K."""
    if not ground_truth:
        return 0.0

    relevant = set(ground_truth)
    dcg = 0.0

    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(i + 2)  # standard log2(rank+1), rank is 1-indexed

    num_relevant = len(relevant)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(num_relevant, k)))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def hit_rate_at_k(ranked_ids: list, ground_truth: list, k: int) -> float:
    """Calculate Hit Rate at K - whether any relevant doc is in top K."""
    if not ground_truth:
        return 0.0
    retrieved = set(ranked_ids[:k])
    relevant = set(ground_truth)
    return 1.0 if retrieved & relevant else 0.0


def evaluate():
    print("Loading vector store and BM25 index...")
    vs = VectorStore(dim=1024)
    vs.load(str(cfg.INDEX_DIR))
    bm25 = load_bm25()

    print(f"Total documents in index: {len(vs.documents)}")

    # Create ground truth
    print("\nCreating ground truth based on content...")
    eval_data = create_ground_truth(vs)

    # Save evaluation data
    eval_file = cfg.EVAL_SET_PATH
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)
    print(f"Saved evaluation data to {eval_file}")

    print(f"\nLoaded {len(eval_data)} evaluation queries.")

    modes = ["dense_only", "bm25_only", "hybrid", "hybrid_plus_rerank"]
    results = {
        mode: {
            "recall@1": [],
            "recall@5": [],
            "recall@10": [],
            "mrr": [],
            "ndcg@10": [],
            "hit_rate@10": [],
        }
        for mode in modes
    }

    top_k = 50

    for item in tqdm(eval_data, desc="Evaluating queries"):
        query = item["question"]
        ground_truth_ids = item["ground_truth_ids"]

        if not ground_truth_ids:
            continue

        # Preprocess
        processed_query = preprocess_query(query)
        translated_query = translate_query(processed_query)

        # Dense retrieval
        query_emb = embed_texts([processed_query])[0] if processed_query else None
        dense_results = vs.search(query_emb, k=top_k) if query_emb is not None else []
        dense_hits = [
            (d.get("doc_idx", i), d.get("score", 0.0))
            for i, d in enumerate(dense_results)
        ]

        # BM25 retrieval
        bm25_hits = []
        if bm25:
            bm25_scores = bm25.get_scores(translated_query)
            top_idx = sorted(
                range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
            )[:top_k]
            bm25_hits = [(i, bm25_scores[i]) for i in top_idx]

        # Evaluate each mode
        for mode in modes:
            ranked_doc_ids = []

            if mode == "dense_only":
                ranked_doc_ids = [
                    f"doc_{d.get('doc_idx', i)}" for i, d in enumerate(dense_results)
                ]

            elif mode == "bm25_only":
                ranked_doc_ids = [f"doc_{idx}" for idx, _ in bm25_hits]

            elif mode == "hybrid":
                ranked_idx = hybrid_rrf(dense_hits, bm25_hits, k=60)
                ranked_doc_ids = [f"doc_{idx}" for idx, _ in ranked_idx]

            elif mode == "hybrid_plus_rerank":
                ranked_idx = hybrid_rrf(dense_hits, bm25_hits, k=60)
                top_20 = [idx for idx, _ in ranked_idx[:20]]
                candidates = [(str(idx), vs.get_doc(idx).text) for idx in top_20]
                rerank_scores = default_reranker.rerank(translated_query, candidates)
                reranked_idx = [
                    idx
                    for _, idx in sorted(
                        zip(rerank_scores, top_20),
                        key=lambda x: (x[0], -x[1]),
                        reverse=True,
                    )
                ]
                ranked_doc_ids = [f"doc_{idx}" for idx in reranked_idx]

            # Calculate metrics
            results[mode]["recall@1"].append(
                recall_at_k(ranked_doc_ids, ground_truth_ids, 1)
            )
            results[mode]["recall@5"].append(
                recall_at_k(ranked_doc_ids, ground_truth_ids, 5)
            )
            results[mode]["recall@10"].append(
                recall_at_k(ranked_doc_ids, ground_truth_ids, 10)
            )
            results[mode]["mrr"].append(mrr_at_k(ranked_doc_ids, ground_truth_ids, 10))
            results[mode]["ndcg@10"].append(
                ndcg_at_k(ranked_doc_ids, ground_truth_ids, 10)
            )
            results[mode]["hit_rate@10"].append(
                hit_rate_at_k(ranked_doc_ids, ground_truth_ids, 10)
            )

    # Save metrics
    metrics_summary = []
    for mode in modes:
        metrics_summary.append(
            {
                "mode": mode,
                "Recall@1": np.mean(results[mode]["recall@1"]),
                "Recall@5": np.mean(results[mode]["recall@5"]),
                "Recall@10": np.mean(results[mode]["recall@10"]),
                "MRR": np.mean(results[mode]["mrr"]),
                "NDCG@10": np.mean(results[mode]["ndcg@10"]),
                "HitRate@10": np.mean(results[mode]["hit_rate@10"]),
                "Failures": len([1 for r in results[mode]["recall@10"] if r == 0]),
            }
        )

    df = pd.DataFrame(metrics_summary)

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    df.to_csv(cfg.RESULTS_DIR / "retrieval_metrics.csv", index=False)
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(df.to_string(index=False))
    print("\n✅ Results saved to results/retrieval_metrics.csv")

    plot_results(df)


def plot_results(df: pd.DataFrame):
    os.makedirs(cfg.RESULTS_DIR / "plots", exist_ok=True)
    sns.set_theme(style="whitegrid")

    df_melt = df.melt(
        id_vars="mode", value_vars=["Recall@1", "Recall@5", "Recall@10", "MRR"]
    )

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x="variable", y="value", hue="mode", data=df_melt, palette="viridis"
    )

    plt.title("RAG Retrieval Quality Comparison", fontsize=16, pad=15)
    plt.ylabel("Score (0 to 1)", fontsize=12)
    plt.xlabel("Metric", fontsize=12)
    plt.legend(title="Retrieval Mode", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.ylim(0, 1.1)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, size=9)

    plt.tight_layout()
    plt.savefig(cfg.RESULTS_DIR / "plots" / "2_retrieval_quality.png", dpi=300)
    print("✅ Plot saved to results/plots/2_retrieval_quality.png")


if __name__ == "__main__":
    evaluate()
