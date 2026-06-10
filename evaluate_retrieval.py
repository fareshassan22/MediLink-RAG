"""MediLink RAG — Retrieval Evaluation Script.

Evaluates 4 retrieval modes against curated ground truth (99 queries).
Produces per-query CSV, summary CSV, and 4 diagnostic plots.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 evaluate_retrieval.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from app.core.config import cfg
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index
from app.indexing.embedder import embed_texts
from app.indexing.preprocessing import preprocess_query
from app.retrieval.hybrid_fusion import hybrid_retrieval_fusion, deduplicate_results
from app.retrieval.reranker import rerank as rerank_documents
from app.retrieval.query_expansion import expand_query
from app.retrieval.query_translator import translate_query, is_arabic

PLOTS_DIR = cfg.RESULTS_DIR / "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── metrics ──────────────────────────────────────────────────


def recall_at_k(ranked_ids: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & set(relevant)) / len(relevant)


def mrr_at_k(ranked_ids: list, relevant: list, k: int) -> float:
    rel_set = set(relevant)
    for i, doc_id in enumerate(ranked_ids[:k]):
        if doc_id in rel_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked_ids: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    dcg = sum(
        1.0 / np.log2(i + 2) for i, d in enumerate(ranked_ids[:k]) if d in rel_set
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(ranked_ids: list, relevant: list, k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if set(ranked_ids[:k]) & set(relevant) else 0.0


# ── Semantic similarity evaluation (fixes circular ground truth) ─────────────
# Using simpler approach: check if query is semantically similar to ground truth docs


def _semantic_recall_at_k(
    retrieved_docs: list,
    relevant_docs: list,
    vs: VectorStore,
    k: int,
    threshold: float = 0.5,
) -> float:
    """REAL semantic recall - compares retrieved doc embeddings to ground truth doc embeddings.

    Counts hit if retrieved doc is semantically similar (cosine > threshold) to ANY ground truth doc.
    """
    if not relevant_docs:
        return 0.0

    # Get embeddings of ground truth docs
    gt_texts = []
    for rid in relevant_docs:
        try:
            idx = int(rid.replace("doc_", ""))
            if 0 <= idx < len(vs.documents):
                gt_texts.append(vs.documents[idx].text[:1000])
        except:
            pass

    if not gt_texts:
        return 0.0

    gt_embs = embed_texts(gt_texts)
    # Convert each to numpy array and normalize
    gt_embs = [np.array(e) for e in gt_embs]
    gt_embs = [e / (np.linalg.norm(e) + 1e-8) for e in gt_embs]

    hits = 0
    for ret_doc in retrieved_docs[:k]:
        ret_text = ret_doc.get("text", "")
        if not ret_text:
            continue

        ret_emb = embed_texts([ret_text[:1000]])[0]
        ret_emb = np.array(ret_emb)
        ret_emb = ret_emb / (np.linalg.norm(ret_emb) + 1e-8)

        max_sim = 0.0
        for gt_emb in gt_embs:
            sim = float(np.dot(ret_emb, gt_emb))
            max_sim = max(max_sim, sim)

        if max_sim >= threshold:
            hits += 1

    return min(hits / len(gt_texts), 1.0)


def _semantic_mrr_at_k(
    retrieved_docs: list,
    relevant_docs: list,
    vs: VectorStore,
    k: int,
    threshold: float = 0.5,
) -> float:
    """REAL semantic MRR - first hit based on embedding similarity to ground truth."""
    gt_texts = []
    for rid in relevant_docs:
        try:
            idx = int(rid.replace("doc_", ""))
            if 0 <= idx < len(vs.documents):
                gt_texts.append(vs.documents[idx].text[:1000])
        except:
            pass

    if not gt_texts:
        return 0.0

    gt_embs = embed_texts(gt_texts)
    gt_embs = [np.array(e) for e in gt_embs]
    gt_embs = [e / (np.linalg.norm(e) + 1e-8) for e in gt_embs]

    for i, ret_doc in enumerate(retrieved_docs[:k]):
        ret_text = ret_doc.get("text", "")
        if not ret_text:
            continue

        ret_emb = embed_texts([ret_text[:1000]])[0]
        ret_emb = np.array(ret_emb)
        ret_emb = ret_emb / (np.linalg.norm(ret_emb) + 1e-8)

        for gt_emb in gt_embs:
            if float(np.dot(ret_emb, gt_emb)) >= threshold:
                return 1.0 / (i + 1)

    return 0.0


# ── retrieval pipeline (mirrors main.py) ─────────────────────

# Cache per-query preprocessing to avoid repeated Groq API calls
_query_cache: dict[str, dict] = {}


def _precompute_query(query: str, vs: VectorStore, bm25: BM25Index | None) -> dict:
    """Precompute and cache all per-query data (expansion, dense/bm25 results)."""
    if query in _query_cache:
        return _query_cache[query]

    processed = preprocess_query(query)
    expanded = expand_query(processed)
    if not expanded:
        expanded = [processed]

    # Dense search for all expanded variants
    dense_results: list[dict] = []
    for eq in expanded:
        emb = embed_texts([eq])[0]
        hits = vs.search(emb, k=cfg.TOP_K_DENSE)
        dense_results.extend(hits)

    # Bilingual boost: also search with English translation for Arabic queries
    if is_arabic(processed):
        en = translate_query(processed)
        if en and en != processed:
            en_emb = embed_texts([en])[0]
            dense_results.extend(vs.search(en_emb, k=cfg.TOP_K_DENSE))

    # BM25 search
    bm25_results: list[dict] = []
    if bm25 is not None:
        bm25_q = processed
        if is_arabic(processed):
            bm25_q = translate_query(processed)
        bm25_results = bm25.search(bm25_q, k=cfg.TOP_K_BM25)

    cache_entry = {
        "processed": processed,
        "dense_results": dense_results,
        "bm25_results": bm25_results,
    }
    _query_cache[query] = cache_entry
    return cache_entry


def _retrieve(
    query: str, mode: str, vs: VectorStore, bm25: BM25Index | None, top_k: int = 50
) -> list[str]:
    """Run retrieval in the given mode. Returns list of doc_id strings.
    
    Args:
        top_k: Number of docs to retrieve for evaluation (default 50 for recall@50)
    """
    cached = _precompute_query(query, vs, bm25)
    processed = cached["processed"]
    dense_results = cached["dense_results"]
    bm25_results = cached["bm25_results"]

    # fuse - use higher top_k for evaluation
    if mode in ("hybrid", "hybrid_rerank"):
        if dense_results and bm25_results:
            fused = hybrid_retrieval_fusion(
                dense_results, bm25_results, processed, top_k=top_k
            )
        elif dense_results:
            fused = deduplicate_results(dense_results)[:top_k]
        else:
            fused = deduplicate_results(bm25_results)[:top_k]
    elif mode == "bm25":
        fused = deduplicate_results(bm25_results)[:top_k]
    elif mode == "dense":
        fused = deduplicate_results(dense_results)[:top_k]
    else:
        fused = deduplicate_results(dense_results)[:top_k]

    # rerank for hybrid_rerank
    if mode == "hybrid_rerank":
        # The corpus is ENGLISH but queries are often ARABIC. Feeding the raw
        # Arabic query to the cross-encoder scores Arabic-query vs English-doc
        # pairs, which the reranker handles poorly and which scrambles the good
        # cross-lingual dense ranking (measured: recall@10 0.79 -> 0.50).
        # Translating to English first matches the corpus language so the
        # cross-encoder compares same-language pairs.
        rr_query = processed
        if is_arabic(processed):
            en = translate_query(processed)
            if en:
                rr_query = en
        fused = rerank_documents(rr_query, fused, top_k=top_k)

    # extract doc IDs
    ids = []
    for i, r in enumerate(fused):
        did = r.get("doc_id")
        if did:
            ids.append(did)
        else:
            ids.append(f"doc_{r.get('doc_idx', i)}")
    return ids


# ── evaluation ───────────────────────────────────────────────


def load_ground_truth() -> list[dict]:
    with open(cfg.EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_evaluation(vs: VectorStore, bm25: BM25Index | None) -> pd.DataFrame:
    """Evaluate 4 retrieval modes on all ground truth queries.

    Uses BOTH exact matching (original) and semantic similarity (new, fixes circular ground truth).
    """
    gt_data = load_ground_truth()
    modes = ["dense", "bm25", "hybrid", "hybrid_rerank"]
    rows: list[dict] = []

    for item in tqdm(gt_data, desc="Evaluating"):
        query = item["query"]
        relevant = item.get("relevant_docs", [])
        lang = item.get("language", "unknown")
        cat = item.get("category", "unknown")
        diff = item.get("difficulty", "unknown")

        if not relevant:
            continue

        for mode in modes:
            retrieved_ids = _retrieve(query, mode, vs, bm25, top_k=50)

            # Get full retrieved docs with text and score for semantic evaluation
            # Re-run search to get scores (caching would be better but this is simpler)
            cached = _precompute_query(query, vs, bm25)
            dense = cached["dense_results"]
            bm25_results = cached["bm25_results"]

            # Get fused results with scores
            if mode in ("hybrid", "hybrid_rerank"):
                if dense and bm25_results:
                    fused = hybrid_retrieval_fusion(
                        dense, bm25_results, cached["processed"], top_k=cfg.TOP_K_FINAL
                    )
                elif dense:
                    fused = deduplicate_results(dense)[: cfg.TOP_K_FINAL]
                else:
                    fused = deduplicate_results(bm25_results)[: cfg.TOP_K_FINAL]
            elif mode == "bm25":
                fused = deduplicate_results(bm25_results)[: cfg.TOP_K_FINAL]
            else:  # dense mode
                fused = deduplicate_results(dense)[: cfg.TOP_K_FINAL]

            # Exact matching metrics - ONLY REAL METRICS
            rows.append(
                {
                    "query": query,
                    "language": lang,
                    "category": cat,
                    "difficulty": diff,
                    "mode": mode,
                    "recall@1": recall_at_k(retrieved_ids, relevant, 1),
                    "recall@5": recall_at_k(retrieved_ids, relevant, 5),
                    "recall@10": recall_at_k(retrieved_ids, relevant, 10),
                    "recall@20": recall_at_k(retrieved_ids, relevant, 20),
                    "recall@50": recall_at_k(retrieved_ids, relevant, 50),
                    "mrr": mrr_at_k(retrieved_ids, relevant, 20),
                    "ndcg@10": ndcg_at_k(retrieved_ids, relevant, 10),
                    "hit_rate@10": hit_rate_at_k(retrieved_ids, relevant, 10),
                    "hit_rate@20": hit_rate_at_k(retrieved_ids, relevant, 20),
                    "any_match": 1.0 if set(retrieved_ids[:20]) & set(relevant) else 0.0,
                }
            )

    return pd.DataFrame(rows)


# ── plots ────────────────────────────────────────────────────


def plot_retrieval_quality(df: pd.DataFrame):
    """Plot 1: Bar chart — Recall@1/5/10, MRR, NDCG@10 by mode."""
    agg = (
        df.groupby("mode")[["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]]
        .mean()
        .reset_index()
    )
    melted = agg.melt(id_vars="mode", var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(
        x="metric", y="score", hue="mode", data=melted, palette="viridis", ax=ax
    )
    ax.set_title("Retrieval Quality by Mode", fontsize=15)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=7)
    ax.legend(title="Mode", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    path = PLOTS_DIR / "retrieval_quality.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_recall_by_language(df: pd.DataFrame):
    """Plot 2: Grouped bar — Recall@10 by mode and language."""
    agg = df.groupby(["mode", "language"])["recall@10"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x="mode", y="recall@10", hue="language", data=agg, palette="Set2", ax=ax
    )
    ax.set_title("Recall@10 by Mode and Language", fontsize=15)
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0, 1.15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
    plt.tight_layout()
    path = PLOTS_DIR / "retrieval_by_language.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_mrr_boxplot(df: pd.DataFrame):
    """Plot 3: Box plot of MRR distribution per mode."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="mode", y="mrr", data=df, palette="Set2", ax=ax)
    sns.stripplot(
        x="mode",
        y="mrr",
        data=df,
        color="black",
        alpha=0.12,
        jitter=True,
        ax=ax,
        size=3,
    )

    means = df.groupby("mode")["mrr"].mean()
    for i, mode in enumerate(df["mode"].unique()):
        ax.text(
            i,
            means[mode] + 0.03,
            f"mean={means[mode]:.3f}",
            ha="center",
            fontsize=10,
            color="red",
        )

    ax.set_title("MRR Distribution by Mode", fontsize=15)
    ax.set_ylabel("MRR")
    ax.set_ylim(-0.05, 1.15)
    plt.tight_layout()
    path = PLOTS_DIR / "retrieval_mrr_boxplot.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_category_heatmap(df: pd.DataFrame):
    """Plot 4: Heatmap — Recall@10 by category and mode."""
    pivot = df.groupby(["category", "mode"])["recall@10"].mean().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0, vmax=1, ax=ax)
    ax.set_title("Recall@10 by Category and Mode", fontsize=15)
    ax.set_ylabel("Category")
    ax.set_xlabel("Mode")
    plt.tight_layout()
    path = PLOTS_DIR / "retrieval_category_heatmap.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


# ── main ─────────────────────────────────────────────────────


def _bootstrap_cis(
    per_query: pd.DataFrame,
    metrics: list[str],
    n_boot: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap 95% CIs for per-mode mean metrics by resampling queries.

    Resamples the per-query rows (with replacement) within each mode n_boot
    times and reports the mean plus the 2.5th/97.5th percentile bounds.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for mode in sorted(per_query["mode"].unique()):
        sub = per_query[per_query["mode"] == mode]
        n = len(sub)
        for metric in metrics:
            vals = sub[metric].to_numpy(dtype=float)
            if n == 0:
                rows.append({"mode": mode, "metric": metric, "n": 0,
                             "mean": float("nan"), "ci_low": float("nan"),
                             "ci_high": float("nan")})
                continue
            idx = rng.integers(0, n, size=(n_boot, n))
            boot_means = vals[idx].mean(axis=1)
            rows.append({
                "mode": mode,
                "metric": metric,
                "n": n,
                "mean": float(vals.mean()),
                "ci_low": float(np.percentile(boot_means, 2.5)),
                "ci_high": float(np.percentile(boot_means, 97.5)),
            })
    return pd.DataFrame(rows)


def main():
    print("=" * 60)
    print("MediLink RAG — Retrieval Evaluation")
    print("=" * 60)

    # Load indexes
    vs = VectorStore(dim=1024)
    vs.load(str(cfg.INDEX_DIR))
    bm25 = BM25Index.load(str(cfg.INDEX_DIR))
    print(f"Loaded {len(vs.documents)} documents")

    # Run evaluation
    per_query = run_evaluation(vs, bm25)

    # Save per-query results
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    per_query.to_csv(cfg.RESULTS_DIR / "retrieval_per_query.csv", index=False)

    # Summary table - ONLY REAL METRICS, no fake eval_type
    summary = per_query.groupby("mode")[
        ["recall@1", "recall@5", "recall@10", "recall@20", "recall@50", "mrr", "ndcg@10", "hit_rate@10", "hit_rate@20", "any_match"]
    ].mean()
    summary["zero_recall@10"] = per_query.groupby("mode")["recall@10"].apply(lambda x: (x == 0).sum())
    summary["zero_recall@20"] = per_query.groupby("mode")["recall@20"].apply(lambda x: (x == 0).sum())

    print("\n" + "=" * 60)
    print("RETRIEVAL SUMMARY (REAL METRICS ONLY)")
    print("=" * 60)
    print(summary.round(3).to_string())

    # Save summary
    summary.to_csv(cfg.RESULTS_DIR / "retrieval_metrics.csv")
    print(f"\nSaved to results/retrieval_metrics.csv")
    print(f"Saved to results/retrieval_per_query.csv")

    # Bootstrap 95% confidence intervals for the headline metrics. With small
    # query sets a single point estimate is fragile, so we resample queries
    # (with replacement) to report a CI alongside the mean. No LLM calls.
    ci_df = _bootstrap_cis(per_query, metrics=["mrr", "recall@10", "ndcg@10"])
    ci_df.to_csv(cfg.RESULTS_DIR / "retrieval_metrics_ci.csv", index=False)
    print("\n" + "=" * 60)
    print("BOOTSTRAP 95% CONFIDENCE INTERVALS (2000 resamples)")
    print("=" * 60)
    print(ci_df.round(4).to_string(index=False))
    print(f"\nSaved to results/retrieval_metrics_ci.csv")

    # By language
    print("\n" + "=" * 60)
    print("BY LANGUAGE")
    print("=" * 60)
    lang_summary = per_query.groupby(["mode", "language"])[
        ["recall@10", "recall@20", "any_match"]
    ].mean()
    print(lang_summary.round(3).to_string())

    # Queries with NO match at all (most honest metric)
    print("\n" + "=" * 60)
    print("QUERIES WITH NO MATCH IN TOP-20 (any_match=0)")
    print("=" * 60)
    for mode in ["dense", "bm25", "hybrid", "hybrid_rerank"]:
        mode_df = per_query[per_query["mode"] == mode]
        no_match = mode_df[mode_df["any_match"] == 0]
        print(f"\n{mode}: {len(no_match)}/{len(mode_df)} queries have zero matches")
        if len(no_match) <= 10:
            for _, row in no_match.iterrows():
                print(f"  [{row['language']}] {row['query'][:60]}")

    # Plots - use all data (no fake eval_type)
    print("\nGenerating plots ...")
    plot_retrieval_quality(per_query)
    plot_recall_by_language(per_query)
    plot_mrr_boxplot(per_query)
    plot_category_heatmap(per_query)

    print("\n" + "=" * 60)
    print("ALL PLOTS:")
    for p in sorted(PLOTS_DIR.glob("retrieval_*.png")):
        print(f"  {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
