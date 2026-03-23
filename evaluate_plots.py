"""
Comprehensive MediLink RAG Evaluation + Plot Generation.

Generates the following plots:
  1. Indexing analysis: text stats before & after preprocessing
  2. Retrieval quality: Recall@1, Recall@5, Recall@10, MRR by mode
  3. Recall@1 & Recall@K distribution (per-query histograms)
  4. Grounding score distribution
  5. Reliability curve (confidence vs grounding)
  6. MRR by retrieval mode (dedicated)
  7. Generation quality: grounding pass/fail rate
"""

from __future__ import annotations

import json
import os
import time
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
from app.indexing.preprocessing import preprocess_query, preprocess_document
from app.retrieval.hybrid_fusion import hybrid_retrieval_fusion, deduplicate_results
from app.retrieval.reranker import rerank as rerank_documents
from app.retrieval.query_expansion import expand_query
from app.retrieval.query_translator import translate_query, is_arabic
from app.safety.judge import judge_answer
from app.generation.prompts import build_prompt

PLOTS_DIR = cfg.RESULTS_DIR / "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Local GPU LLM (avoids Groq rate limits) ─────────────────

_local_pipeline = None


def _get_local_llm():
    global _local_pipeline
    if _local_pipeline is not None:
        return _local_pipeline
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    print(f"  Loading {model_name} across GPU 0+1 (bf16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        max_memory={0: "47GiB", 1: "47GiB"},
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    _local_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )
    print(f"  ✅ {model_name} loaded")
    return _local_pipeline


def local_generate(prompt: str) -> str:
    pipe = _get_local_llm()
    messages = [
        {"role": "system", "content": "You are a medical AI assistant. Answer accurately based only on the provided context. Be concise."},
        {"role": "user", "content": prompt},
    ]
    out = pipe(messages, max_new_tokens=512, do_sample=True, temperature=0.2, top_p=0.9)
    return out[0]["generated_text"][-1]["content"]

# ── helpers ──────────────────────────────────────────────────


def _load_ground_truth() -> list[dict]:
    with open(cfg.EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _doc_id_from_result(result: dict, fallback_idx: int) -> str:
    """Extract doc_id string from a retrieval result dict."""
    did = result.get("doc_id")
    if did:
        return did
    idx = result.get("doc_idx", fallback_idx)
    return f"doc_{idx}"


# ── 1. Indexing analysis ─────────────────────────────────────


def plot_indexing_analysis(vs: VectorStore):
    """Analyse text before and after preprocessing; plot distributions."""
    print("\n[1/7] Indexing Analysis ...")

    raw_lengths = []
    processed_lengths = []
    raw_word_counts = []
    processed_word_counts = []
    vocab_raw: set[str] = set()
    vocab_processed: set[str] = set()

    for doc in vs.documents:
        raw = doc.text
        processed = preprocess_document(raw)

        raw_lengths.append(len(raw))
        processed_lengths.append(len(processed))
        raw_word_counts.append(len(raw.split()))
        processed_word_counts.append(len(processed.split()))
        vocab_raw.update(raw.split()[:50])
        vocab_processed.update(processed.split()[:50])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("1. Indexing: Text Before vs After Preprocessing", fontsize=16, y=1.02)

    # char length distribution
    axes[0, 0].hist(raw_lengths, bins=40, alpha=0.6, label="Raw", color="#2196F3")
    axes[0, 0].hist(processed_lengths, bins=40, alpha=0.6, label="Processed", color="#4CAF50")
    axes[0, 0].set_title("Character Length Distribution")
    axes[0, 0].set_xlabel("Characters")
    axes[0, 0].set_ylabel("Chunks")
    axes[0, 0].legend()

    # word count distribution
    axes[0, 1].hist(raw_word_counts, bins=40, alpha=0.6, label="Raw", color="#2196F3")
    axes[0, 1].hist(processed_word_counts, bins=40, alpha=0.6, label="Processed", color="#4CAF50")
    axes[0, 1].set_title("Word Count Distribution")
    axes[0, 1].set_xlabel("Words")
    axes[0, 1].set_ylabel("Chunks")
    axes[0, 1].legend()

    # summary stats bar chart
    stats_labels = ["Avg Chars", "Avg Words", "Vocab Size (sample)"]
    raw_stats = [np.mean(raw_lengths), np.mean(raw_word_counts), len(vocab_raw)]
    proc_stats = [np.mean(processed_lengths), np.mean(processed_word_counts), len(vocab_processed)]
    x = np.arange(len(stats_labels))
    w = 0.35
    axes[1, 0].bar(x - w / 2, raw_stats, w, label="Raw", color="#2196F3")
    axes[1, 0].bar(x + w / 2, proc_stats, w, label="Processed", color="#4CAF50")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(stats_labels)
    axes[1, 0].set_title("Summary Statistics")
    axes[1, 0].legend()
    for i, (rv, pv) in enumerate(zip(raw_stats, proc_stats)):
        axes[1, 0].text(i - w / 2, rv + 1, f"{rv:.0f}", ha="center", fontsize=8)
        axes[1, 0].text(i + w / 2, pv + 1, f"{pv:.0f}", ha="center", fontsize=8)

    # chunks per page
    page_counts: dict[int, int] = {}
    for doc in vs.documents:
        p = doc.metadata.get("page", 0)
        page_counts[p] = page_counts.get(p, 0) + 1
    pages_sorted = sorted(page_counts.keys())
    axes[1, 1].bar(pages_sorted, [page_counts[p] for p in pages_sorted], color="#FF9800", width=1.0)
    axes[1, 1].set_title("Chunks per Page")
    axes[1, 1].set_xlabel("Page Number")
    axes[1, 1].set_ylabel("Chunks")

    plt.tight_layout()
    path = PLOTS_DIR / "1_indexing_analysis.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {path}")


# ── 2-3. Retrieval evaluation ────────────────────────────────


def _retrieve(query: str, mode: str, vs: VectorStore, bm25: BM25Index | None):
    """Run retrieval in the given mode. Returns list of doc_id strings."""
    processed = preprocess_query(query)
    expanded = expand_query(processed)
    if not expanded:
        expanded = [processed]

    dense_results: list[dict] = []
    bm25_results: list[dict] = []

    if mode in ("dense", "hybrid", "hybrid_rerank"):
        for eq in expanded:
            emb = embed_texts([eq])[0]
            hits = vs.search(emb, k=10)
            dense_results.extend(hits)

    if mode in ("bm25", "hybrid", "hybrid_rerank"):
        if bm25 is not None:
            bm25_q = processed
            if is_arabic(processed):
                bm25_q = translate_query(processed)
            bm25_results = bm25.search(bm25_q, k=10)

    # fuse
    if mode == "hybrid" or mode == "hybrid_rerank":
        if dense_results and bm25_results:
            fused = hybrid_retrieval_fusion(dense_results, bm25_results, processed, top_k=10)
        elif dense_results:
            fused = deduplicate_results(dense_results)[:10]
        else:
            fused = deduplicate_results(bm25_results)[:10]
    elif mode == "bm25":
        fused = deduplicate_results(bm25_results)[:10]
    else:
        fused = deduplicate_results(dense_results)[:10]

    # rerank for hybrid_rerank
    if mode == "hybrid_rerank":
        fused = rerank_documents(processed, fused, top_k=10)

    return [_doc_id_from_result(r, i) for i, r in enumerate(fused)]


def run_retrieval_eval(vs: VectorStore, bm25: BM25Index | None):
    """Evaluate 4 retrieval modes, return per-query metrics DataFrame."""
    print("\n[2/7] Retrieval Evaluation ...")
    gt_data = _load_ground_truth()
    modes = ["dense", "bm25", "hybrid", "hybrid_rerank"]

    rows: list[dict] = []

    for item in tqdm(gt_data, desc="  Evaluating"):
        query = item["query"]
        relevant = item.get("relevant_docs", [])
        lang = item.get("language", "unknown")
        cat = item.get("category", "unknown")
        diff = item.get("difficulty", "unknown")

        if not relevant:
            continue

        for mode in modes:
            retrieved = _retrieve(query, mode, vs, bm25)
            relevant_set = set(relevant)

            # per-query recall@k
            def _recall(k):
                return len(set(retrieved[:k]) & relevant_set) / len(relevant_set) if relevant_set else 0.0

            def _mrr():
                for i, d in enumerate(retrieved):
                    if d in relevant_set:
                        return 1.0 / (i + 1)
                return 0.0

            rows.append({
                "query": query,
                "language": lang,
                "category": cat,
                "difficulty": diff,
                "mode": mode,
                "recall@1": _recall(1),
                "recall@5": _recall(5),
                "recall@10": _recall(10),
                "mrr": _mrr(),
            })

    return pd.DataFrame(rows)


def plot_retrieval_quality(df: pd.DataFrame):
    """Plot 2: Retrieval quality bar chart (Recall@1/5/10, MRR) by mode."""
    print("\n[3/7] Retrieval Quality Bar Chart ...")
    agg = df.groupby("mode")[["recall@1", "recall@5", "recall@10", "mrr"]].mean().reset_index()
    melted = agg.melt(id_vars="mode", var_name="metric", value_name="score")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x="metric", y="score", hue="mode", data=melted, palette="viridis", ax=ax)
    ax.set_title("2. Retrieval Quality by Mode", fontsize=15)
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 1.1)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
    ax.legend(title="Mode", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    path = PLOTS_DIR / "2_retrieval_quality.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {path}")


def plot_recall_distributions(df: pd.DataFrame):
    """Plot 3: Recall@1 and Recall@5 per-query distribution (histogram)."""
    print("\n[4/7] Recall Distribution Histograms ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, metric in enumerate(["recall@1", "recall@5"]):
        for mode in df["mode"].unique():
            subset = df[df["mode"] == mode][metric]
            axes[i].hist(subset, bins=20, alpha=0.45, label=mode)
        axes[i].set_title(f"3{chr(97+i)}. {metric} Distribution (per query)")
        axes[i].set_xlabel(metric)
        axes[i].set_ylabel("Queries")
        axes[i].legend()

    plt.tight_layout()
    path = PLOTS_DIR / "3_recall_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {path}")


def plot_mrr_by_mode(df: pd.DataFrame):
    """Plot 6: MRR by mode — box plot + mean."""
    print("\n[5/7] MRR by Mode ...")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x="mode", y="mrr", data=df, palette="Set2", ax=ax)
    sns.stripplot(x="mode", y="mrr", data=df, color="black", alpha=0.15, jitter=True, ax=ax, size=3)

    means = df.groupby("mode")["mrr"].mean()
    for i, mode in enumerate(df["mode"].unique()):
        ax.text(i, means[mode] + 0.03, f"μ={means[mode]:.3f}", ha="center", fontsize=10, color="red")

    ax.set_title("6. MRR Distribution by Retrieval Mode", fontsize=15)
    ax.set_ylabel("MRR")
    ax.set_ylim(-0.05, 1.15)
    plt.tight_layout()
    path = PLOTS_DIR / "6_mrr_by_mode.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {path}")


# ── 4-5. End-to-end generation eval (grounding + confidence) ─


def run_generation_eval(vs: VectorStore, bm25: BM25Index | None):
    """Run full pipeline (retrieve → generate → ground) on a sample of queries.
    Returns DataFrame with per-query grounding, confidence, mode."""
    print("\n[6/7] End-to-End Generation Evaluation ...")
    gt_data = _load_ground_truth()
    modes = ["dense", "hybrid", "hybrid_rerank"]

    # Use ALL queries — local GPU has no rate limits
    sample = gt_data
    rows: list[dict] = []

    # pre-load local LLM
    _get_local_llm()

    for item in tqdm(sample, desc="  Generating"):
        query = item["query"]
        lang = item.get("language", "unknown")
        cat = item.get("category", "unknown")

        for mode in modes:
            # retrieve
            processed = preprocess_query(query)
            expanded = expand_query(processed)
            if not expanded:
                expanded = [processed]

            dense_results: list[dict] = []
            bm25_results: list[dict] = []

            if mode in ("dense", "hybrid", "hybrid_rerank"):
                for eq in expanded:
                    emb = embed_texts([eq])[0]
                    hits = vs.search(emb, k=10)
                    dense_results.extend(hits)

            if mode in ("hybrid", "hybrid_rerank"):
                if bm25 is not None:
                    bm25_q = processed
                    if is_arabic(processed):
                        bm25_q = translate_query(processed)
                    bm25_results = bm25.search(bm25_q, k=10)

            # fuse
            if mode in ("hybrid", "hybrid_rerank") and dense_results and bm25_results:
                fused = hybrid_retrieval_fusion(dense_results, bm25_results, processed, top_k=10)
            else:
                fused = deduplicate_results(dense_results)[:10]

            if mode == "hybrid_rerank":
                fused = rerank_documents(processed, fused, top_k=10)

            # top chunks — dynamic selection with lower threshold
            top_chunks = [c for c in fused if c.get("rerank_score_normalized", c.get("dense_score", 0)) > 0.25][:10]
            if not top_chunks:
                top_chunks = fused[:3]
            context_texts = [c.get("text", "") for c in top_chunks if c.get("text")]
            context = "\n\n".join(context_texts[:3])

            # generate locally on GPU (no rate limits)
            answer = ""
            try:
                prompt = build_prompt(query, context, "patient")
                answer = local_generate(prompt)
            except Exception as e:
                print(f"  [WARN] Generation failed: {e}")

            # judge (LLM-based grounding + hallucination + confidence)
            judge_result = judge_answer(
                query=query,
                answer=answer,
                context_texts=context_texts,
            )
            grounded = judge_result.grounded
            g_score = judge_result.grounding_score
            confidence = round(max(0.0, min(0.95, judge_result.confidence)), 3)

            # retrieval score (for reference)
            retrieval_scores = []
            for d in top_chunks:
                s = d.get("rerank_score_normalized", d.get("dense_score", 0.0))
                retrieval_scores.append(max(0.0, min(1.0, s)))
            top_3 = sorted(retrieval_scores, reverse=True)[:3]
            ret_score = sum(top_3) / len(top_3) if top_3 else 0.0

            rows.append({
                "query": query,
                "language": lang,
                "category": cat,
                "mode": mode,
                "grounding_score": round(g_score, 3),
                "grounded": grounded,
                "confidence": confidence,
                "retrieval_score": round(ret_score, 3),
                "answer_len": len(answer),
            })

    return pd.DataFrame(rows)


def plot_grounding_distribution(df: pd.DataFrame):
    """Plot 4: Grounding score distribution."""
    print("\n  Plotting grounding distribution ...")
    fig, ax = plt.subplots(figsize=(10, 5))
    for mode in df["mode"].unique():
        subset = df[df["mode"] == mode]["grounding_score"]
        ax.hist(subset, bins=15, alpha=0.5, label=mode)
    ax.axvline(x=0.30, color="red", linestyle="--", label="Threshold (0.30)")
    ax.set_title("4. Grounding Score Distribution", fontsize=15)
    ax.set_xlabel("Grounding Score")
    ax.set_ylabel("Queries")
    ax.legend()
    plt.tight_layout()
    path = PLOTS_DIR / "4_grounding_distribution.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {path}")


def plot_reliability_curve(df: pd.DataFrame):
    """Plot 5: Reliability / calibration curve — confidence vs grounding."""
    print("\n  Plotting reliability curve ...")
    n_bins = 8
    confidences = df["confidence"].values
    groundings = df["grounding_score"].values

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accs.append(groundings[mask].mean())
            bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # top: reliability curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.plot(bin_centers, bin_accs, "o-", color="#E53935", markersize=8, label="Model")
    ax1.fill_between(bin_centers, bin_accs, [c for c in bin_centers], alpha=0.15, color="red")
    ax1.set_ylabel("Mean Grounding Score")
    ax1.set_title("5. Reliability Curve: Confidence vs Grounding", fontsize=15)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # bottom: histogram of confidence values
    ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins * 0.8, color="#42A5F5", alpha=0.7)
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")

    plt.tight_layout()
    path = PLOTS_DIR / "5_reliability_curve.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {path}")


def plot_generation_quality(df: pd.DataFrame):
    """Plot 7: Grounding pass/fail rate by mode."""
    print("\n  Plotting generation quality ...")
    grouped = df.groupby("mode")["grounded"].value_counts(normalize=True).unstack(fill_value=0)
    plot_df = grouped.reindex(columns=[True, False], fill_value=0)
    plot_df.columns = ["Grounded", "Not Grounded"]

    fig, ax = plt.subplots(figsize=(9, 5))
    plot_df.plot(kind="bar", stacked=True, ax=ax,
                 color=["#4CAF50", "#E53935"], edgecolor="white")
    ax.set_title("7. Generation Quality: Grounding Pass/Fail Rate", fontsize=15)
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Retrieval Mode")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="center", fontsize=10)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path = PLOTS_DIR / "7_generation_quality.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✅ Saved {path}")


# ── main ─────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("MediLink RAG — Comprehensive Evaluation")
    print("=" * 60)

    # load indexes
    vs = VectorStore(dim=1024)
    vs.load(str(cfg.INDEX_DIR))
    bm25 = BM25Index.load(str(cfg.INDEX_DIR))
    print(f"Loaded {len(vs.documents)} documents")

    # ── Plot 1: Indexing analysis ──
    plot_indexing_analysis(vs)

    # ── Plots 2, 3, 6: Retrieval ──
    retrieval_df = run_retrieval_eval(vs, bm25)
    retrieval_df.to_csv(cfg.RESULTS_DIR / "retrieval_per_query.csv", index=False)
    print(f"  Saved per-query retrieval metrics to results/retrieval_per_query.csv")

    plot_retrieval_quality(retrieval_df)
    plot_recall_distributions(retrieval_df)
    plot_mrr_by_mode(retrieval_df)

    # print summary table
    summary = retrieval_df.groupby("mode")[["recall@1", "recall@5", "recall@10", "mrr"]].mean()
    print("\n" + "=" * 60)
    print("RETRIEVAL SUMMARY")
    print("=" * 60)
    print(summary.to_string())

    # ── Plots 4, 5, 7: Generation ──
    gen_df = run_generation_eval(vs, bm25)
    gen_df.to_csv(cfg.RESULTS_DIR / "generation_eval.csv", index=False)
    print(f"\n  Saved generation eval to results/generation_eval.csv")

    plot_grounding_distribution(gen_df)
    plot_reliability_curve(gen_df)
    plot_generation_quality(gen_df)

    # final summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    gen_summary = gen_df.groupby("mode")[["grounding_score", "confidence"]].mean()
    print(gen_summary.to_string())

    print("\n" + "=" * 60)
    print("ALL PLOTS:")
    for p in sorted(PLOTS_DIR.glob("*.png")):
        print(f"  📊 {p}")
    print("=" * 60)


if __name__ == "__main__":
    main()
