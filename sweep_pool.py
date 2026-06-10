"""Honest sweep of TOON_POOL_SIZE (rerank candidate-pool depth) on the FROZEN GT.

Pool 30/50 approximate the previous fixed-pool behaviour; 0 = whole patient
corpus. Same metric code as the main evaluator; GT untouched.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

import json
from pathlib import Path

import pandas as pd

from app.core.config import cfg
from app.evaluation.metrics import recall_at_k, ndcg_at_k, mrr

QUERY_SET = Path("data/toon_multipatient_queries.json")
GT_PATH = Path("data/toon_rowlevel_ground_truth_multipatient.json")

POOLS = ["30", "50", "0"]  # 0 = whole corpus


def capped_recall(ranked_lists, gt_lists, k):
    vals = []
    for r, g in zip(ranked_lists, gt_lists):
        gs = set(g)
        if not gs:
            continue
        vals.append(len(set(r[:k]) & gs) / min(k, len(gs)))
    return sum(vals) / max(1, len(vals))


def hit_at_k(ranked_lists, gt_lists, k):
    vals = []
    for r, g in zip(ranked_lists, gt_lists):
        gs = set(g)
        if not gs:
            continue
        vals.append(1.0 if set(r[:k]) & gs else 0.0)
    return sum(vals) / max(1, len(vals))


def main():
    queries = json.load(open(QUERY_SET, encoding="utf-8"))
    gt = {r["query"]: r.get("relevant_ids", []) for r in json.load(open(GT_PATH, encoding="utf-8"))}

    from app.retrieval.toon import search_bm25, search_hybrid, load_patient_index, index_patient

    tier_int = {"tier_1_simple": 1, "tier_2_moderate": 2, "tier_3_complex": 3}
    for pid in sorted({q["patient_id"] for q in queries}):
        vs, bm25 = load_patient_index(pid)
        if not vs and not bm25:
            index_patient(pid)

    meta = [{
        "query": q["query"], "patient_id": q["patient_id"],
        "tier": tier_int[q["tier"]], "category": q.get("category") or "uncategorized",
        "relevant_ids": gt.get(q["query"], []),
    } for q in queries]

    summary_rows = []
    per_pool_cat = {}
    for pool in POOLS:
        os.environ["TOON_POOL_SIZE"] = pool
        recs = []
        for m in meta:
            if m["tier"] == 1:
                ranked = search_bm25(m["patient_id"], m["query"], top_k=10, return_ids=True)
            else:
                ranked = search_hybrid(m["patient_id"], m["query"], top_k=10, return_ids=True)
            recs.append({**m, "ranked_ids": [r for r in ranked if r]})
        df = pd.DataFrame(recs)
        scored = df[df["relevant_ids"].map(len) > 0]
        rl = list(scored["ranked_ids"]); gl = list(scored["relevant_ids"])
        label = "full" if pool == "0" else pool
        summary_rows.append({
            "pool": label, "n": len(scored),
            "recall@1": round(recall_at_k(rl, gl, 1), 4),
            "recall@5": round(recall_at_k(rl, gl, 5), 4),
            "recall@10": round(recall_at_k(rl, gl, 10), 4),
            "capR@5": round(capped_recall(rl, gl, 5), 4),
            "capR@10": round(capped_recall(rl, gl, 10), 4),
            "hit@1": round(hit_at_k(rl, gl, 1), 4),
            "hit@3": round(hit_at_k(rl, gl, 3), 4),
            "mrr": round(mrr(rl, gl), 4),
            "ndcg@10": round(ndcg_at_k(rl, gl, 10), 4),
        })
        per_pool_cat[label] = {
            c: round(capped_recall(list(scored[scored["category"] == c]["ranked_ids"]),
                                   list(scored[scored["category"] == c]["relevant_ids"]), 5), 4)
            for c in sorted(scored["category"].unique())
        }
        print(f"  done pool={label}")

    sdf = pd.DataFrame(summary_rows)
    print("\n=== OVERALL by pool depth (blend=0, frozen GT) ===")
    print(sdf.to_string(index=False))

    print("\n=== capped_recall@5 by category, by pool ===")
    labels = [("full" if p == "0" else p) for p in POOLS]
    cats = sorted({c for d in per_pool_cat.values() for c in d})
    print("category".ljust(14) + "".join(f"{l:>8}" for l in labels))
    for c in cats:
        print(c.ljust(14) + "".join(f"{per_pool_cat[l].get(c, 0):>8}" for l in labels))

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out = Path(cfg.RESULTS_DIR) / f"toon_pool_sweep_{ts}.csv"
    sdf.to_csv(out, index=False)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
