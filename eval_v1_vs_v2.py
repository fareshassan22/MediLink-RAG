"""Honest before/after on v1 vs v2 GT.

Same retrieval code, same metric code, only the GT file changes.
Reports overall + per-tier + per-category, plus a focused report on the
4 queries that the v2 builder modified.
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
GT_V1 = Path("data/toon_rowlevel_ground_truth_multipatient.json")
GT_V2 = Path("data/toon_rowlevel_ground_truth_multipatient_v2.json")


def capped_recall(rl, gl, k):
    out = []
    for r, g in zip(rl, gl):
        gs = set(g)
        if not gs:
            continue
        out.append(len(set(r[:k]) & gs) / min(k, len(gs)))
    return sum(out) / max(1, len(out))


def hit_at_k(rl, gl, k):
    out = []
    for r, g in zip(rl, gl):
        gs = set(g)
        if not gs:
            continue
        out.append(1.0 if set(r[:k]) & gs else 0.0)
    return sum(out) / max(1, len(out))


def precision_at_1(rl, gl):
    out = []
    for r, g in zip(rl, gl):
        if not g:
            continue
        out.append(1.0 if r and r[0] in set(g) else 0.0)
    return sum(out) / max(1, len(out))


def evaluate(gt_path: Path, ranked_by_query: dict) -> dict:
    gt_list = json.loads(gt_path.read_text(encoding="utf-8"))
    gt_by_q = {e["query"]: e.get("relevant_ids", []) for e in gt_list}
    rl, gl = [], []
    per_tier = {1: [[], []], 2: [[], []], 3: [[], []]}
    per_cat: dict = {}
    tier_int = {"tier_1_simple": 1, "tier_2_moderate": 2, "tier_3_complex": 3}
    for q in queries:
        g = gt_by_q.get(q["query"], [])
        if not g:
            continue
        r = ranked_by_query[q["query"]]
        rl.append(r); gl.append(g)
        t = tier_int[q["tier"]]
        per_tier[t][0].append(r); per_tier[t][1].append(g)
        cat = q.get("category") or "uncategorized"
        per_cat.setdefault(cat, [[], []])
        per_cat[cat][0].append(r); per_cat[cat][1].append(g)

    out = {
        "n": len(rl),
        "P@1": round(precision_at_1(rl, gl), 4),
        "recall@1": round(recall_at_k(rl, gl, 1), 4),
        "recall@5": round(recall_at_k(rl, gl, 5), 4),
        "recall@10": round(recall_at_k(rl, gl, 10), 4),
        "capR@5": round(capped_recall(rl, gl, 5), 4),
        "hit@1": round(hit_at_k(rl, gl, 1), 4),
        "hit@5": round(hit_at_k(rl, gl, 5), 4),
        "mrr": round(mrr(rl, gl), 4),
        "ndcg@10": round(ndcg_at_k(rl, gl, 10), 4),
    }
    out["_per_tier"] = {
        t: {
            "n": len(per_tier[t][0]),
            "recall@5": round(recall_at_k(*per_tier[t], 5), 4),
            "capR@5": round(capped_recall(*per_tier[t], 5), 4),
            "hit@1": round(hit_at_k(*per_tier[t], 1), 4),
            "mrr": round(mrr(*per_tier[t]), 4),
        }
        for t in (1, 2, 3) if per_tier[t][0]
    }
    out["_per_cat"] = {
        c: {
            "n": len(per_cat[c][0]),
            "capR@5": round(capped_recall(*per_cat[c], 5), 4),
            "hit@1": round(hit_at_k(*per_cat[c], 1), 4),
        }
        for c in sorted(per_cat)
    }
    return out


# ─── Main ────────────────────────────────────────────────────────────────────

queries = json.loads(QUERY_SET.read_text(encoding="utf-8"))
tier_int = {"tier_1_simple": 1, "tier_2_moderate": 2, "tier_3_complex": 3}

from app.retrieval.toon import search_bm25, search_hybrid, load_patient_index, index_patient

# warm caches
for pid in sorted({q["patient_id"] for q in queries}):
    vs, bm25 = load_patient_index(pid)
    if not vs and not bm25:
        index_patient(pid)

# Run retrieval ONCE — same ranked lists used to score against both v1 & v2.
ranked_by_query: dict[str, list[str]] = {}
for q in queries:
    if tier_int[q["tier"]] == 1:
        r = search_bm25(q["patient_id"], q["query"], top_k=10, return_ids=True)
    else:
        r = search_hybrid(q["patient_id"], q["query"], top_k=10, return_ids=True)
    ranked_by_query[q["query"]] = [x for x in r if x]
print(f"ran retrieval on {len(ranked_by_query)} queries")

m_v1 = evaluate(GT_V1, ranked_by_query)
m_v2 = evaluate(GT_V2, ranked_by_query)

print("\n" + "=" * 66)
print("OVERALL  (same retrieval, same metric code, only GT changes)")
print("=" * 66)
print(f"  metric       v1 (frozen)       v2 (single-answer scoped)   delta")
for k in ["P@1", "recall@1", "recall@5", "recall@10", "capR@5", "hit@1", "hit@5", "mrr", "ndcg@10"]:
    a, b = m_v1[k], m_v2[k]
    arrow = "↑" if b > a else ("↓" if b < a else "·")
    print(f"  {k:<12} {a:<17} {b:<27} {arrow}{(b-a):+.4f}")

print("\n=== per-tier (v1 vs v2) ===")
for t in (1, 2, 3):
    a = m_v1["_per_tier"].get(t); b = m_v2["_per_tier"].get(t)
    if not a:
        continue
    print(f"  T{t} (n={a['n']}): recall@5 {a['recall@5']}->{b['recall@5']}  "
          f"capR@5 {a['capR@5']}->{b['capR@5']}  hit@1 {a['hit@1']}->{b['hit@1']}  "
          f"mrr {a['mrr']}->{b['mrr']}")

print("\n=== focus on the 4 v2-modified queries (where we expect the biggest swing) ===")
v2_entries = json.loads(GT_V2.read_text(encoding="utf-8"))
v1_entries = {e["query"]: e for e in json.loads(GT_V1.read_text(encoding="utf-8"))}
modified = [e for e in v2_entries if e.get("_v2_modified")]
print(f"  {len(modified)} queries modified")
for e in modified:
    q = e["query"]
    r = ranked_by_query[q]
    g_v1 = set(v1_entries[q].get("relevant_ids", []))
    g_v2 = set(e["relevant_ids"])
    h1_v1 = 1 if r and r[0] in g_v1 else 0
    h1_v2 = 1 if r and r[0] in g_v2 else 0
    h5_v1 = int(bool(set(r[:5]) & g_v1))
    h5_v2 = int(bool(set(r[:5]) & g_v2))
    print(f"\n  [{e['_v2_rule']}] '{q}'")
    print(f"    v1 GT size: {len(g_v1)}    v2 GT size: {len(g_v2)}")
    print(f"    rank@1 in v1? {h1_v1}    in v2? {h1_v2}")
    print(f"    hit@5 v1: {h5_v1}        hit@5 v2: {h5_v2}")
    print(f"    top-3 retrieved: {r[:3]}")
    print(f"    v2 relevant:     {sorted(g_v2)}")

print("\nSaved v2 ranked lists embedded in this run; full eval CSVs unchanged.")
