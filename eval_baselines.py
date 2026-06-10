"""
Retrieval ablation / baselines table.

Compares the full routed system against component-ablated baselines, on the
SAME 100 queries and the SAME strict v2 GT, so every number is comparable.

Variants
--------
  bm25_only         BM25 ranking only. No cross-encoder rerank. No router.
  dense_only        Dense (bge-m3) ranking only. No rerank. No router.
  hybrid_no_rerank  RRF(dense, bm25). No cross-encoder rerank. No router.
  hybrid_no_router  Full hybrid (RRF + cross-encoder) on EVERY query (router off).
  full_system       Production routing: T1->BM25+rerank, T2/T3->hybrid+rerank.

All variants retrieve top_k=10 so every @k (k<=10) is computable.

Statistics
----------
  * 95% bootstrap confidence intervals on MRR for every variant.
  * Paired bootstrap significance test (full_system vs each baseline) on MRR,
    using the SAME resampled query indices for both systems (paired design).

Outputs
-------
  results/baselines_{ts}.csv               summary metrics + MRR 95% CI per variant
  results/baselines_sig_{ts}.csv           paired tests: full_system vs each baseline
  results/baselines_per_query_{ts}.csv     per-query reciprocal rank for every variant
  results/plots/baselines_mrr_{ts}.png     MRR bar chart with 95% CI error bars
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
os.environ.setdefault("TOON_ROUTER_MODE", "hybrid")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.core.config import cfg
from app.evaluation.metrics import recall_at_k, precision_at_k, ndcg_at_k, mrr

QUERIES = Path("data/toon_multipatient_queries.json")
GT      = Path("data/toon_rowlevel_ground_truth_multipatient_v2.json")
TIER_INT = {"tier_1_simple": 1, "tier_2_moderate": 2, "tier_3_complex": 3}

RESULTS = Path(cfg.RESULTS_DIR)
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)

N_BOOT = 2000           # bootstrap resamples
SEED   = 1234           # deterministic resampling


# ─── Retrieval variants — each returns a ranked list of doc_ids ──────────────

def _ids_from(results):
    return [r["metadata"].get("doc_id", "") for r in results
            if r.get("metadata") and r["metadata"].get("doc_id")]


def retrieve_bm25_only(pid, query, vs, bm25, top_k=10):
    if not bm25:
        return []
    return _ids_from(bm25.search(query, k=top_k))


def retrieve_dense_only(pid, query, vs, bm25, top_k=10):
    if not vs:
        return []
    from app.indexing.embedder import embed_texts
    emb = embed_texts([query])[0]
    return _ids_from(vs.search(emb, k=top_k))


def retrieve_hybrid_no_rerank(pid, query, vs, bm25, top_k=10):
    """RRF fusion of dense + bm25, NO cross-encoder rerank."""
    if not vs and not bm25:
        return []
    from app.indexing.embedder import embed_texts
    rrf_k = 60
    scores: dict[str, float] = {}
    text_to_id: dict[str, str] = {}

    pool_each = max(top_k * 3, 30)
    if vs:
        emb = embed_texts([query])[0]
        for rank, d in enumerate(vs.search(emb, k=pool_each)):
            t = d.get("text")
            if not t:
                continue
            text_to_id[t] = d.get("metadata", {}).get("doc_id", "")
            scores[t] = scores.get(t, 0.0) + 1.0 / (rrf_k + rank + 1)
    if bm25:
        for rank, d in enumerate(bm25.search(query, k=pool_each)):
            t = d.get("text")
            if not t:
                continue
            text_to_id.setdefault(t, d.get("metadata", {}).get("doc_id", ""))
            scores[t] = scores.get(t, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [text_to_id.get(t, "") for t, _ in ranked if text_to_id.get(t)]


def retrieve_hybrid_no_router(pid, query, vs, bm25, top_k=10):
    """Full hybrid (RRF + cross-encoder) applied to EVERY query — router off."""
    from app.retrieval.toon import search_hybrid
    return [r for r in search_hybrid(pid, query, top_k=top_k, return_ids=True) if r]


def retrieve_full_system(pid, query, vs, bm25, tier, top_k=10):
    """Production routing: T1 -> BM25+rerank, T2/T3 -> hybrid+rerank."""
    from app.retrieval.toon import search_bm25, search_hybrid
    if tier == 1:
        return [r for r in search_bm25(pid, query, top_k=top_k, return_ids=True) if r]
    return [r for r in search_hybrid(pid, query, top_k=top_k, return_ids=True) if r]


VARIANTS = [
    ("bm25_only",        retrieve_bm25_only),
    ("dense_only",       retrieve_dense_only),
    ("hybrid_no_rerank", retrieve_hybrid_no_rerank),
    ("hybrid_no_router", retrieve_hybrid_no_router),
    # full_system handled separately (needs tier)
]


# ─── Metrics ─────────────────────────────────────────────────────────────────

def hit_at_k(rl, gl, k):
    v = []
    for r, g in zip(rl, gl):
        gs = set(g)
        if not gs:
            continue
        v.append(1.0 if set(r[:k]) & gs else 0.0)
    return sum(v) / max(1, len(v))


def per_query_rr(ranked, relevant):
    rel = set(relevant)
    for i, d in enumerate(ranked, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def metrics_block(ranked_lists, relevant_lists):
    return {
        "recall@1":    round(recall_at_k(ranked_lists, relevant_lists, 1), 4),
        "recall@5":    round(recall_at_k(ranked_lists, relevant_lists, 5), 4),
        "recall@10":   round(recall_at_k(ranked_lists, relevant_lists, 10), 4),
        "precision@5": round(precision_at_k(ranked_lists, relevant_lists, 5), 4),
        "hit@10":      round(hit_at_k(ranked_lists, relevant_lists, 10), 4),
        "ndcg@10":     round(ndcg_at_k(ranked_lists, relevant_lists, 10), 4),
        "mrr":         round(mrr(ranked_lists, relevant_lists), 4),
    }


def bootstrap_ci(values, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI for the mean of a per-query metric vector."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = arr[idx].mean()
    lo = np.percentile(means, 100 * alpha / 2)
    hi = np.percentile(means, 100 * (1 - alpha / 2))
    return float(arr.mean()), float(lo), float(hi)


def paired_bootstrap_test(rr_a, rr_b, n_boot=N_BOOT, seed=SEED):
    """Paired bootstrap on per-query reciprocal-rank vectors.

    Returns (mean_diff = A-B, two-sided p-value, ci_lo, ci_hi).
    Same resampled indices applied to both systems (paired design).
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(rr_a, dtype=float)
    b = np.asarray(rr_b, dtype=float)
    n = len(a)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    mean_diff = float(a.mean() - b.mean())
    # two-sided p: proportion of resamples on the opposite side of 0
    p = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    p = float(min(1.0, p))
    lo = float(np.percentile(diffs, 2.5))
    hi = float(np.percentile(diffs, 97.5))
    return mean_diff, p, lo, hi


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    gt = {e["query"]: e.get("relevant_ids", [])
          for e in json.loads(GT.read_text(encoding="utf-8"))}
    print(f"loaded {len(queries)} queries, GT: {GT.name}", flush=True)

    from app.retrieval.toon import load_patient_index, index_patient

    print("warming patient indexes …", flush=True)
    for pid in sorted({q["patient_id"] for q in queries}):
        vs, bm = load_patient_index(pid)
        if not vs and not bm:
            index_patient(pid)

    # Keep only queries that have non-empty GT (scored set), preserve order.
    scored = [q for q in queries if gt.get(q["query"])]
    print(f"scoring {len(scored)}/{len(queries)} queries (non-empty GT)", flush=True)
    relevant_lists = [gt[q["query"]] for q in scored]

    # Run every variant. ranked_by_variant[name] = list aligned with `scored`.
    ranked_by_variant: dict[str, list] = {}
    t0 = time.time()

    for name, fn in VARIANTS:
        print(f"running variant: {name} …", flush=True)
        ranked = []
        for q in scored:
            vs, bm25 = load_patient_index(q["patient_id"])
            ranked.append(fn(q["patient_id"], q["query"], vs, bm25, top_k=10))
        ranked_by_variant[name] = ranked
        print(f"  done {name} ({time.time()-t0:.1f}s)", flush=True)

    # full_system (tier-routed)
    print("running variant: full_system …", flush=True)
    ranked = []
    for q in scored:
        vs, bm25 = load_patient_index(q["patient_id"])
        ranked.append(retrieve_full_system(
            q["patient_id"], q["query"], vs, bm25, TIER_INT[q["tier"]], top_k=10))
    ranked_by_variant["full_system"] = ranked
    print(f"  done full_system ({time.time()-t0:.1f}s)", flush=True)

    variant_order = [n for n, _ in VARIANTS] + ["full_system"]

    # Per-query reciprocal rank for every variant (for bootstrap + paired tests).
    rr_by_variant = {
        name: [per_query_rr(r, rel)
               for r, rel in zip(ranked_by_variant[name], relevant_lists)]
        for name in variant_order
    }

    # Summary table with MRR 95% CI.
    summary_rows = []
    for name in variant_order:
        block = metrics_block(ranked_by_variant[name], relevant_lists)
        mrr_mean, mrr_lo, mrr_hi = bootstrap_ci(rr_by_variant[name])
        summary_rows.append({
            "variant": name,
            "n": len(scored),
            **block,
            "mrr_ci_lo": round(mrr_lo, 4),
            "mrr_ci_hi": round(mrr_hi, 4),
        })
    sdf = pd.DataFrame(summary_rows)

    # Paired significance: full_system vs each baseline (MRR).
    sig_rows = []
    base_rr = rr_by_variant["full_system"]
    for name in variant_order:
        if name == "full_system":
            continue
        diff, p, lo, hi = paired_bootstrap_test(base_rr, rr_by_variant[name])
        sig_rows.append({
            "comparison": f"full_system - {name}",
            "mrr_diff": round(diff, 4),
            "ci95_lo": round(lo, 4),
            "ci95_hi": round(hi, 4),
            "p_value": round(p, 4),
            "significant_0.05": "yes" if p < 0.05 else "no",
        })
    sig_df = pd.DataFrame(sig_rows)

    # Per-query RR dump.
    pq_rows = []
    for i, q in enumerate(scored):
        row = {"query": q["query"], "patient_id": q["patient_id"],
               "tier": TIER_INT[q["tier"]], "n_relevant": len(relevant_lists[i])}
        for name in variant_order:
            row[f"rr_{name}"] = round(rr_by_variant[name][i], 4)
        pq_rows.append(row)
    pq_df = pd.DataFrame(pq_rows)

    ts = time.strftime("%Y%m%d_%H%M%S")
    sout  = RESULTS / f"baselines_{ts}.csv"
    sigout = RESULTS / f"baselines_sig_{ts}.csv"
    pqout = RESULTS / f"baselines_per_query_{ts}.csv"
    sdf.to_csv(sout, index=False)
    sig_df.to_csv(sigout, index=False)
    pq_df.to_csv(pqout, index=False)

    print("\n" + "=" * 88)
    print(f"BASELINES / ABLATION  (GT: {GT.name}, n={len(scored)}, {N_BOOT} bootstrap)")
    print("=" * 88)
    print(sdf.to_string(index=False))
    print("\nPAIRED SIGNIFICANCE — full_system vs each baseline (MRR):")
    print(sig_df.to_string(index=False))
    print(f"\nsaved: {sout}")
    print(f"saved: {sigout}")
    print(f"saved: {pqout}")

    # Plot — MRR with 95% CI error bars.
    fig, ax = plt.subplots(figsize=(10, 5))
    order = variant_order
    means = [sdf.loc[sdf.variant == v, "mrr"].iloc[0] for v in order]
    los   = [sdf.loc[sdf.variant == v, "mrr_ci_lo"].iloc[0] for v in order]
    his   = [sdf.loc[sdf.variant == v, "mrr_ci_hi"].iloc[0] for v in order]
    yerr = np.array([[m - lo for m, lo in zip(means, los)],
                     [hi - m for m, hi in zip(means, his)]])
    colors = ["#888"] * (len(order) - 1) + ["#2a7"]
    ax.bar(order, means, yerr=yerr, capsize=6, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("MRR")
    ax.set_title("Retrieval ablation — MRR with 95% bootstrap CI (strict v2 GT)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    pplot = PLOTS / f"baselines_mrr_{ts}.png"
    plt.savefig(pplot, dpi=120); plt.close()
    print(f"saved: {pplot}")


if __name__ == "__main__":
    main()
