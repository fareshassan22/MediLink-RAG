"""
Focused retrieval evaluation — only the metrics requested:
  recall@1, recall@5, recall@10, ndcg@10, hit@10, mrr, precision@5

Uses strict v2 GT, hybrid router (production default), 100-query set.
Outputs:
  results/retrieval_focused_{ts}.csv          (overall + per-tier + per-cat)
  results/retrieval_focused_per_query_{ts}.csv
  results/plots/retrieval_focused_{ts}.png
"""
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
os.environ.setdefault("TOON_ROUTER_MODE", "hybrid")

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


def hit_at_k(rl, gl, k):
    v = []
    for r, g in zip(rl, gl):
        gs = set(g)
        if not gs: continue
        v.append(1.0 if set(r[:k]) & gs else 0.0)
    return sum(v) / max(1, len(v))


def metrics_block(sub_df: pd.DataFrame) -> dict:
    rl = list(sub_df["ranked_ids"])
    gl = list(sub_df["relevant_ids"])
    return {
        "n":          len(sub_df),
        "recall@1":   round(recall_at_k(rl, gl, 1), 4),
        "recall@5":   round(recall_at_k(rl, gl, 5), 4),
        "recall@10":  round(recall_at_k(rl, gl, 10), 4),
        "precision@5":round(precision_at_k(rl, gl, 5), 4),
        "hit@10":     round(hit_at_k(rl, gl, 10), 4),
        "ndcg@10":    round(ndcg_at_k(rl, gl, 10), 4),
        "mrr":        round(mrr(rl, gl), 4),
    }


def main():
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    gt = {e["query"]: e.get("relevant_ids", []) for e in json.loads(GT.read_text(encoding="utf-8"))}
    print(f"loaded {len(queries)} queries, GT: {GT.name}", flush=True)

    from app.retrieval.toon import (
        search_bm25, search_hybrid, load_patient_index, index_patient,
    )

    print("warming patient indexes …", flush=True)
    for pid in sorted({q["patient_id"] for q in queries}):
        vs, bm = load_patient_index(pid)
        if not vs and not bm:
            index_patient(pid)

    # Retrieve once for each query at top_k=10 (covers all @k up to 10).
    print("running retrieval …", flush=True)
    rows = []
    t0 = time.time()
    for i, q in enumerate(queries, 1):
        t = TIER_INT[q["tier"]]
        pid = q["patient_id"]
        if t == 1:
            ranked = search_bm25(pid, q["query"], top_k=10, return_ids=True)
        else:
            ranked = search_hybrid(pid, q["query"], top_k=10, return_ids=True)
        rows.append({
            "query": q["query"],
            "patient_id": pid,
            "tier": t,
            "category": q.get("category") or "uncategorized",
            "ranked_ids": [r for r in ranked if r],
            "relevant_ids": gt.get(q["query"], []),
            "n_relevant": len(gt.get(q["query"], [])),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(queries)}  ({time.time()-t0:.1f}s)", flush=True)
    df = pd.DataFrame(rows)
    scored = df[df["n_relevant"] > 0]
    print(f"scored {len(scored)}/{len(df)} queries (others have empty GT)", flush=True)

    # Overall
    overall = metrics_block(scored)
    # Per tier
    per_tier = []
    for tier in (1, 2, 3):
        sub = scored[scored["tier"] == tier]
        if not len(sub): continue
        per_tier.append({"slice": f"T{tier}", **metrics_block(sub)})
    # Per category
    per_cat = []
    for cat in sorted(scored["category"].unique()):
        sub = scored[scored["category"] == cat]
        if not len(sub): continue
        per_cat.append({"slice": cat, **metrics_block(sub)})

    summary_rows = [{"slice": "OVERALL", **overall}] + per_tier + per_cat
    sdf = pd.DataFrame(summary_rows)

    ts = time.strftime("%Y%m%d_%H%M%S")
    sout = RESULTS / f"retrieval_focused_{ts}.csv"
    pout = RESULTS / f"retrieval_focused_per_query_{ts}.csv"
    sdf.to_csv(sout, index=False)
    df.drop(columns=["ranked_ids", "relevant_ids"]).to_csv(pout, index=False)

    print("\n" + "=" * 80)
    print(f"FOCUSED RETRIEVAL METRICS  (GT: {GT.name}, hybrid router)")
    print("=" * 80)
    print(sdf.to_string(index=False))
    print(f"\nsaved: {sout}")
    print(f"saved: {pout}")

    # Plot — per-tier bars of the 7 metrics
    fig, ax = plt.subplots(figsize=(11, 5))
    metric_cols = ["recall@1", "recall@5", "recall@10", "precision@5", "hit@10", "ndcg@10", "mrr"]
    tier_rows = [r for r in summary_rows if r["slice"].startswith("T")]
    if tier_rows:
        plot_df = pd.DataFrame(tier_rows).melt(
            id_vars="slice", value_vars=metric_cols, var_name="metric", value_name="score"
        )
        sns.barplot(data=plot_df, x="metric", y="score", hue="slice", ax=ax)
        ax.set_ylim(0, 1.0)
        ax.set_title("TOON retrieval metrics by tier (strict v2 GT)")
        ax.set_xlabel(""); ax.set_ylabel("score")
        plt.xticks(rotation=20)
        plt.tight_layout()
        pplot = PLOTS / f"retrieval_focused_tier_{ts}.png"
        plt.savefig(pplot, dpi=120); plt.close()
        print(f"saved: {pplot}")

    # Plot — per-category MRR / hit@10 / nDCG@10
    if per_cat:
        cat_df = pd.DataFrame(per_cat)
        fig, ax = plt.subplots(figsize=(11, 5))
        plot_df = cat_df.melt(
            id_vars="slice", value_vars=["hit@10", "mrr", "ndcg@10"], var_name="metric", value_name="score"
        )
        sns.barplot(data=plot_df, x="slice", y="score", hue="metric", ax=ax)
        ax.set_ylim(0, 1.0)
        ax.set_title("TOON retrieval by category (strict v2 GT)")
        ax.set_xlabel("")
        plt.xticks(rotation=20)
        plt.tight_layout()
        pplot = PLOTS / f"retrieval_focused_cat_{ts}.png"
        plt.savefig(pplot, dpi=120); plt.close()
        print(f"saved: {pplot}")


if __name__ == "__main__":
    main()
