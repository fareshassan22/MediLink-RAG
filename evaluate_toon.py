"""MediLink TOON — Comprehensive Evaluation Script.

Evaluates the TOON 3-tier patient retrieval system the same way the generic
RAG system is evaluated (Recall/NDCG/MRR vs ground truth), plus two TOON-only
layers the generic RAG eval has no equivalent for.

This script evaluates the TOON pipeline ONLY — it never touches the generic
RAG corpus (data/processed). Every metric is produced by TOON's own code path:
the TOON router and TOON's per-patient retrieval/tier functions.

THREE LAYERS
────────────
1. TOON retrieval & token budget — For each query, run TOON's actual tier
                         retrieval (search_bm25 / search_hybrid / fetch_live_context
                         from app.retrieval.toon) against the patient's own index.
                         Measure non-empty retrieval rate and whether the retrieved
                         context respects each tier's TOKEN_BUDGETS. Needs Supabase
                         + the embedder. (TOON returns text, not corpus doc_ids, so
                         classic Recall@K/NDCG against the global corpus does NOT
                         apply to the TOON path — that would measure the RAG system.)
2. Router accuracy     — TOON-only. Runs classify() on every query and compares
                         the predicted tier with the expected tier (the group the
                         query lives in inside TOON_TEST_QUERIES). Pure regex —
                         needs NO GPU, NO Supabase, NO LLM. Always runnable.
3. End-to-end quality  — Runs each query through PatientRAGService.run() and
                         records grounding_score, confidence, latency and the
                         tier actually used. Needs Supabase + the embedder.

Each layer is independent and degrades gracefully: if the patient index or
Supabase is unavailable, that layer is skipped and the others still run.

Outputs (in results/):
  toon_router_confusion_{ts}.csv     — predicted vs expected tier matrix
  toon_router_per_query_{ts}.csv     — per-query routing decisions
  toon_retrieval_{ts}.csv            — per-query TOON retrieval + token budget
  toon_retrieval_summary_{ts}.csv    — per-tier retrieval/budget summary
  toon_endtoend_{ts}.csv             — grounding/latency/tier per query
  toon_eval_summary_{ts}.json        — everything combined
  plots/toon_*.png                   — confusion matrix + budget/latency charts

Usage:
    python3 evaluate_toon.py                       # all available layers
    python3 evaluate_toon.py --layers router       # only router accuracy (no GPU)
    python3 evaluate_toon.py --layers router,retrieval,endtoend
    python3 evaluate_toon.py --mode quick          # 5 queries per tier
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from app.core.config import cfg
from app.evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mrr,
    grounding_rate,
)
from app.retrieval.toon_router import classify
from app.retrieval.toon import TOKEN_BUDGETS
from tests.test_rag_queries import TOON_TEST_QUERIES

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text or ""))
except Exception:  # tiktoken unavailable — fall back to a rough word estimate
    def _count_tokens(text: str) -> int:
        return len((text or "").split())

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("evaluate_toon")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(cfg.RESULTS_DIR)
PLOTS_DIR = RESULTS_DIR / "plots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

# TOON_TEST_QUERIES group name -> integer tier label
_TIER_OF_GROUP = {
    "tier_1_simple": 1,
    "tier_2_moderate": 2,
    "tier_3_complex": 3,
}

# Row-level ground truth produced by build_toon_rowlevel_gt.py
ROWLEVEL_GT_PATH = cfg.DATA_DIR / "toon_rowlevel_ground_truth.json"

# Optional multi-patient query set + matching ground truth (override via env).
_QUERY_SET = os.environ.get("TOON_QUERY_SET", "")
if _QUERY_SET:
    ROWLEVEL_GT_PATH = Path(
        os.environ.get("TOON_GT_OUTPUT", cfg.DATA_DIR / "toon_rowlevel_ground_truth_multipatient.json")
    )


# ─── Query loading ───────────────────────────────────────────────────────────

# tier string -> integer
_TIER_INT = {"tier_1_simple": 1, "tier_2_moderate": 2, "tier_3_complex": 3}


def load_queries(mode: str = "full") -> List[Dict]:
    """Flatten queries into [{query, expected_tier, group, patient_id}].

    Uses the multi-patient query set when TOON_QUERY_SET points at a JSON file
    of [{"query","tier","patient_id"}]; otherwise the single-patient
    TOON_TEST_QUERIES.
    """
    if _QUERY_SET and os.path.exists(_QUERY_SET):
        with open(_QUERY_SET, "r", encoding="utf-8") as f:
            raw = json.load(f)
        out: List[Dict] = []
        for item in raw:
            out.append(
                {
                    "query": item["query"],
                    "expected_tier": _TIER_INT[item["tier"]],
                    "group": item["tier"],
                    "patient_id": int(item["patient_id"]),
                    "category": item.get("category"),
                }
            )
        if mode == "quick":
            out = out[:15]
        return out

    out = []
    for group, data in TOON_TEST_QUERIES.items():
        expected_tier = _TIER_OF_GROUP[group]
        patient_id = data["queries"]["patient_id"]
        questions = data["queries"]["questions"]
        if mode == "quick":
            questions = questions[:5]
        for q in questions:
            out.append(
                {
                    "query": q,
                    "expected_tier": expected_tier,
                    "group": group,
                    "patient_id": patient_id,
                }
            )
    return out


# ─── Layer 2 — Router accuracy (no external deps) ────────────────────────────

def evaluate_router(queries: List[Dict]) -> tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Classify each query and compare predicted vs expected tier."""
    print(f"\n{'='*60}\nLAYER 2 — Router Accuracy\n{'='*60}")

    per_query = []
    for item in queries:
        decision = classify(item["query"], item["patient_id"])
        per_query.append(
            {
                "query": item["query"],
                "expected_tier": item["expected_tier"],
                "predicted_tier": decision.tier,
                "correct": int(decision.tier == item["expected_tier"]),
                "reason": decision.reason,
                "language": decision.language,
            }
        )

    df = pd.DataFrame(per_query)

    # Confusion matrix (rows = expected, cols = predicted)
    tiers = [1, 2, 3]
    matrix = pd.DataFrame(0, index=tiers, columns=tiers)
    for _, row in df.iterrows():
        matrix.loc[row["expected_tier"], row["predicted_tier"]] += 1
    matrix.index.name = "expected"
    matrix.columns.name = "predicted"

    accuracy = float(df["correct"].mean()) if len(df) else 0.0

    # Per-tier precision/recall from the confusion matrix
    per_tier_stats = {}
    for t in tiers:
        tp = int(matrix.loc[t, t])
        expected_total = int(matrix.loc[t].sum())            # actually tier t
        predicted_total = int(matrix[t].sum())               # predicted tier t
        recall = tp / expected_total if expected_total else 0.0
        precision = tp / predicted_total if predicted_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        per_tier_stats[f"tier_{t}"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": expected_total,
        }

    summary = {"accuracy": round(accuracy, 4), "per_tier": per_tier_stats}

    print(f"  Overall routing accuracy: {accuracy:.1%}")
    for t in tiers:
        s = per_tier_stats[f"tier_{t}"]
        print(f"  Tier {t}: precision={s['precision']:.2f} recall={s['recall']:.2f} "
              f"f1={s['f1']:.2f} (n={s['support']})")

    return df, matrix, summary


# ─── Layer 1 — TOON retrieval & token-budget adherence ───────────────────────

def evaluate_retrieval(queries: List[Dict]) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Run TOON's OWN tier retrieval per query and score it on its own terms.

    For each query we call the exact function TOON uses for that tier:
        tier 1 -> search_bm25        (top_k=5)
        tier 2 -> search_hybrid      (top_k=10)
        tier 3 -> fetch_live_context (full record)
    then measure:
        - retrieved        : was any context returned (non-empty)?
        - tokens           : token count of the retrieved context
        - budget           : TOKEN_BUDGETS[tier]
        - within_budget    : tokens <= budget

    This is TOON-only: it never touches the global RAG corpus.
    """
    print(f"\n{'='*60}\nLAYER 1 — TOON Retrieval & Token Budget\n{'='*60}")

    # Lazy import — pulls Supabase client + embedder
    try:
        from app.retrieval.toon import (
            search_bm25,
            search_hybrid,
            fetch_live_context,
            load_patient_index,
            index_patient,
        )
    except Exception as e:
        print(f"  SKIPPED — could not import TOON retrieval: {type(e).__name__}: {e}")
        return None, None

    # Ensure each patient referenced by the queries has an index built once.
    patient_ids = sorted({item["patient_id"] for item in queries})
    for pid in patient_ids:
        try:
            vs, bm25 = load_patient_index(pid)
            if not vs and not bm25:
                print(f"  Indexing patient {pid} …")
                n = index_patient(pid)
                if n == 0:
                    print(f"  WARNING — no data indexed for patient {pid} "
                          f"(Supabase empty/unreachable)")
        except Exception as e:
            print(f"  SKIPPED — could not build patient index: {type(e).__name__}: {e}")
            return None, None

    def _retrieve(item: Dict) -> str:
        tier, q, pid = item["expected_tier"], item["query"], item["patient_id"]
        if tier == 1:
            return search_bm25(pid, q, top_k=5, token_budget=TOKEN_BUDGETS[1])
        if tier == 2:
            return search_hybrid(pid, q, top_k=10, token_budget=TOKEN_BUDGETS[2])
        return fetch_live_context(pid)

    per_query = []
    for item in queries:
        try:
            context = _retrieve(item)
        except Exception as e:
            logger.warning("TOON retrieval failed for %r: %s", item["query"][:40], e)
            context = ""
        tier = item["expected_tier"]
        tokens = _count_tokens(context)
        budget = TOKEN_BUDGETS[tier]
        per_query.append(
            {
                "query": item["query"],
                "tier": tier,
                "retrieved": int(bool(context)),
                "tokens": tokens,
                "budget": budget,
                "within_budget": int(tokens <= budget),
            }
        )

    pq_df = pd.DataFrame(per_query)

    rows = []
    for tier in sorted(pq_df["tier"].unique()):
        sub = pq_df[pq_df["tier"] == tier]
        rows.append(
            {
                "tier": int(tier),
                "n_queries": len(sub),
                "budget": TOKEN_BUDGETS[tier],
                "retrieval_rate": round(float(sub["retrieved"].mean()), 4),
                "avg_tokens": round(float(sub["tokens"].mean()), 1),
                "max_tokens": int(sub["tokens"].max()),
                "within_budget_rate": round(float(sub["within_budget"].mean()), 4),
            }
        )

    summary_df = pd.DataFrame(rows)
    print(summary_df.to_string(index=False))
    return summary_df, pq_df


# ─── Layer 1b — Retrieval METRICS vs row-level ground truth ───────────────────

def evaluate_metrics(queries: List[Dict]) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Recall@K / Precision@K / NDCG@K / MRR per tier for TOON's own retrieval.

    Uses the row-level ground truth (stable doc_ids) from
    build_toon_rowlevel_gt.py and TOON's own search functions with
    return_ids=True. Tier 1 -> search_bm25, Tier 2 -> search_hybrid. Tier 3
    fetches the whole record, so ranked-retrieval metrics don't apply there.

    Also reports capped Recall@K (recall / min(k, n_relevant)) and Hit@K
    (>=1 relevant in top-k), and a per-category breakdown.
    """
    print(f"\n{'='*60}\nLAYER 1b — Retrieval Metrics (Recall/NDCG/MRR)\n{'='*60}")

    if not ROWLEVEL_GT_PATH.exists():
        print(f"  SKIPPED — row-level ground truth not found at {ROWLEVEL_GT_PATH}")
        print("  Generate it first: CUDA_VISIBLE_DEVICES=0 python3 build_toon_rowlevel_gt.py")
        return None, None

    with open(ROWLEVEL_GT_PATH, "r", encoding="utf-8") as f:
        gt_map = {r["query"]: r.get("relevant_ids", []) for r in json.load(f)}

    try:
        from app.retrieval.toon import search_bm25, search_hybrid, load_patient_index, index_patient
    except Exception as e:
        print(f"  SKIPPED — could not import TOON retrieval: {type(e).__name__}: {e}")
        return None, None

    for pid in sorted({item["patient_id"] for item in queries}):
        vs, bm25 = load_patient_index(pid)
        if not vs and not bm25:
            index_patient(pid)

    per_query = []
    for item in queries:
        q, tier, pid = item["query"], item["expected_tier"], item["patient_id"]
        relevant = gt_map.get(q, [])
        # Tier 3 grabs the full record — selective-retrieval metrics are N/A.
        if tier == 1:
            ranked = search_bm25(pid, q, top_k=10, return_ids=True)
        else:
            ranked = search_hybrid(pid, q, top_k=10, return_ids=True)
        per_query.append(
            {
                "query": q,
                "tier": tier,
                "category": item.get("category") or "uncategorized",
                "n_relevant": len(relevant),
                "ranked_ids": [r for r in ranked if r],
                "relevant_ids": relevant,
            }
        )

    pq_df = pd.DataFrame(per_query)

    def _capped_recall(ranked_lists, gt_lists, k):
        vals = []
        for ranked_l, gt_l in zip(ranked_lists, gt_lists):
            gt_set = set(gt_l)
            if not gt_set:
                continue
            hits = len(set(ranked_l[:k]) & gt_set)
            vals.append(hits / min(k, len(gt_set)))
        return sum(vals) / max(1, len(vals))

    def _hit_at_k(ranked_lists, gt_lists, k):
        vals = []
        for ranked_l, gt_l in zip(ranked_lists, gt_lists):
            gt_set = set(gt_l)
            if not gt_set:
                continue
            vals.append(1.0 if set(ranked_l[:k]) & gt_set else 0.0)
        return sum(vals) / max(1, len(vals))

    def _metric_block(sub: pd.DataFrame) -> Dict:
        ranked_lists = list(sub["ranked_ids"])
        gt_lists = list(sub["relevant_ids"])
        block: Dict = {}
        for k in (1, 3, 5, 10):
            block[f"recall@{k}"] = round(recall_at_k(ranked_lists, gt_lists, k), 4)
            block[f"capped_recall@{k}"] = round(_capped_recall(ranked_lists, gt_lists, k), 4)
            block[f"hit@{k}"] = round(_hit_at_k(ranked_lists, gt_lists, k), 4)
            block[f"precision@{k}"] = round(precision_at_k(ranked_lists, gt_lists, k), 4)
            block[f"ndcg@{k}"] = round(ndcg_at_k(ranked_lists, gt_lists, k), 4)
        block["mrr"] = round(mrr(ranked_lists, gt_lists), 4)
        return block

    rows = []
    # Only tiers 1 & 2 use selective retrieval; report both but flag tier 3.
    for tier in sorted(pq_df["tier"].unique()):
        sub = pq_df[(pq_df["tier"] == tier) & (pq_df["n_relevant"] > 0)]
        if not len(sub):
            rows.append({"tier": int(tier), "n_scored": 0})
            continue
        row = {"tier": int(tier), "n_scored": len(sub)}
        row.update(_metric_block(sub))
        rows.append(row)

    metrics_df = pd.DataFrame(rows)
    print(metrics_df.to_string(index=False))

    # Per-category breakdown (across all tiers)
    cat_rows = []
    for cat in sorted(pq_df["category"].unique()):
        sub = pq_df[(pq_df["category"] == cat) & (pq_df["n_relevant"] > 0)]
        if not len(sub):
            cat_rows.append({"category": cat, "n_scored": 0})
            continue
        row = {"category": cat, "n_scored": len(sub)}
        row.update(_metric_block(sub))
        cat_rows.append(row)
    cat_df = pd.DataFrame(cat_rows)
    print(f"\n--- Per-category breakdown ---")
    print(cat_df.to_string(index=False))

    return metrics_df, pq_df, cat_df


# ─── Layer 3 — End-to-end grounding / latency ────────────────────────────────

def evaluate_endtoend(queries: List[Dict]) -> Optional[pd.DataFrame]:
    """Run the full TOON pipeline and record grounding/confidence/latency/tier."""
    print(f"\n{'='*60}\nLAYER 3 — End-to-End Quality (grounding + latency)\n{'='*60}")

    try:
        from app.retrieval.toon_service import patient_rag_service
    except Exception as e:
        print(f"  SKIPPED — could not import service: {type(e).__name__}: {e}")
        return None

    rows = []
    for item in queries:
        q = item["query"]
        start = time.time()
        try:
            res = patient_rag_service.run(query=q, patient_id=item["patient_id"])
            rows.append(
                {
                    "query": q,
                    "expected_tier": item["expected_tier"],
                    "tier_used": res.tier_used,
                    "status": res.status,
                    "grounding_score": res.grounding_score,
                    "confidence": res.confidence,
                    "latency_seconds": round(time.time() - start, 3),
                }
            )
        except Exception as e:
            logger.warning("run failed for %r: %s", q[:40], e)
            rows.append(
                {
                    "query": q,
                    "expected_tier": item["expected_tier"],
                    "tier_used": -1,
                    "status": f"error:{type(e).__name__}",
                    "grounding_score": 0.0,
                    "confidence": 0.0,
                    "latency_seconds": round(time.time() - start, 3),
                }
            )
        print(f"  [{item['group']}] {q[:40]}... → tier={rows[-1]['tier_used']} "
              f"grounding={rows[-1]['grounding_score']} ({rows[-1]['latency_seconds']}s)")

    df = pd.DataFrame(rows)
    ok = df[df["tier_used"] > 0]
    if len(ok):
        print(f"\n  Avg grounding: {ok['grounding_score'].mean():.3f}")
        print(f"  Grounding rate (>=0.5): {grounding_rate(list(ok['grounding_score'])):.1%}")
        print(f"  Avg latency: {ok['latency_seconds'].mean():.2f}s")
    return df


# ─── Plots ───────────────────────────────────────────────────────────────────

def plot_router_confusion(matrix: pd.DataFrame, ts: str) -> None:
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=True)
    plt.title("TOON Router — Expected vs Predicted Tier")
    plt.ylabel("Expected tier")
    plt.xlabel("Predicted tier")
    plt.tight_layout()
    path = PLOTS_DIR / f"toon_router_confusion_{ts}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved {path}")


def plot_retrieval_budget(summary_df: pd.DataFrame, ts: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(data=summary_df, x="tier", y="retrieval_rate", ax=axes[0], color="#4c72b0")
    axes[0].set_title("TOON Retrieval Rate by Tier")
    axes[0].set_ylim(0, 1)
    sns.barplot(data=summary_df, x="tier", y="within_budget_rate", ax=axes[1], color="#55a868")
    axes[1].set_title("Token-Budget Adherence by Tier")
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    path = PLOTS_DIR / f"toon_retrieval_budget_{ts}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved {path}")


def plot_retrieval_metrics(metrics_df: pd.DataFrame, ts: str) -> None:
    cols = [c for c in ("recall@5", "ndcg@5", "mrr") if c in metrics_df.columns]
    scored = metrics_df[metrics_df.get("n_scored", 0) > 0] if "n_scored" in metrics_df else metrics_df
    if not len(scored) or not cols:
        return
    plot_df = scored.melt(id_vars="tier", value_vars=cols, var_name="metric", value_name="score")
    plt.figure(figsize=(8, 5))
    sns.barplot(data=plot_df, x="metric", y="score", hue="tier")
    plt.title("TOON Retrieval Quality by Tier (row-level GT)")
    plt.ylim(0, 1)
    plt.tight_layout()
    path = PLOTS_DIR / f"toon_retrieval_metrics_{ts}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved {path}")


def plot_latency_by_tier(e2e_df: pd.DataFrame, ts: str) -> None:
    ok = e2e_df[e2e_df["tier_used"] > 0]
    if not len(ok):
        return
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=ok, x="tier_used", y="latency_seconds")
    plt.title("TOON Latency by Tier Used")
    plt.xlabel("Tier")
    plt.ylabel("Latency (s)")
    plt.tight_layout()
    path = PLOTS_DIR / f"toon_latency_by_tier_{ts}.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  saved {path}")


# ─── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the TOON 3-tier system")
    parser.add_argument(
        "--layers",
        default="router,retrieval,metrics,endtoend",
        help="Comma-separated subset of: router, retrieval, metrics, endtoend",
    )
    parser.add_argument("--mode", choices=["full", "quick"], default="full")
    args = parser.parse_args()

    layers = {x.strip() for x in args.layers.split(",") if x.strip()}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'#'*60}\n# MediLink TOON Evaluation\n# layers={sorted(layers)} mode={args.mode}\n# {datetime.now().isoformat()}\n{'#'*60}")

    queries = load_queries(args.mode)
    print(f"Loaded {len(queries)} TOON queries")

    summary: Dict = {"timestamp": ts, "mode": args.mode, "n_queries": len(queries)}

    # Layer 2 — Router (always available)
    if "router" in layers:
        router_df, matrix, router_summary = evaluate_router(queries)
        router_df.to_csv(RESULTS_DIR / f"toon_router_per_query_{ts}.csv", index=False)
        matrix.to_csv(RESULTS_DIR / f"toon_router_confusion_{ts}.csv")
        summary["router"] = router_summary
        plot_router_confusion(matrix, ts)

    # Layer 1 — TOON retrieval & token budget
    if "retrieval" in layers:
        summary_df, pq_df = evaluate_retrieval(queries)
        if summary_df is not None:
            summary_df.to_csv(RESULTS_DIR / f"toon_retrieval_summary_{ts}.csv", index=False)
            pq_df.to_csv(RESULTS_DIR / f"toon_retrieval_{ts}.csv", index=False)
            summary["retrieval"] = summary_df.to_dict(orient="records")
            plot_retrieval_budget(summary_df, ts)

    # Layer 1b — Retrieval metrics vs row-level ground truth
    if "metrics" in layers:
        metrics_df, mpq_df, cat_df = evaluate_metrics(queries)
        if metrics_df is not None:
            metrics_df.to_csv(RESULTS_DIR / f"toon_retrieval_metrics_{ts}.csv", index=False)
            mpq_df.drop(columns=["ranked_ids", "relevant_ids"]).to_csv(
                RESULTS_DIR / f"toon_metrics_per_query_{ts}.csv", index=False
            )
            if cat_df is not None:
                cat_df.to_csv(RESULTS_DIR / f"toon_retrieval_metrics_by_category_{ts}.csv", index=False)
            summary["metrics"] = metrics_df.to_dict(orient="records")
            plot_retrieval_metrics(metrics_df, ts)

    # Layer 3 — End-to-end
    if "endtoend" in layers:
        e2e_df = evaluate_endtoend(queries)
        if e2e_df is not None:
            e2e_df.to_csv(RESULTS_DIR / f"toon_endtoend_{ts}.csv", index=False)
            ok = e2e_df[e2e_df["tier_used"] > 0]
            summary["endtoend"] = {
                "avg_grounding": round(float(ok["grounding_score"].mean()), 4) if len(ok) else None,
                "grounding_rate": round(grounding_rate(list(ok["grounding_score"])), 4) if len(ok) else None,
                "avg_latency": round(float(ok["latency_seconds"].mean()), 4) if len(ok) else None,
                "n_success": int(len(ok)),
            }
            plot_latency_by_tier(e2e_df, ts)

    with open(RESULTS_DIR / f"toon_eval_summary_{ts}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}\nDONE — results saved to {RESULTS_DIR}\n{'='*60}")


if __name__ == "__main__":
    main()
