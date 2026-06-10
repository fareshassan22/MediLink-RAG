#!/usr/bin/env python3
"""Generate ONE consolidated TOON evaluation CSV.

Pulls together every TOON metric into a single tidy (long-format) CSV:
  - Router accuracy: original vs regex-fixed vs learned vs hybrid, test & held-out
  - Retrieval IR metrics per tier (Recall/Precision/NDCG/MRR) from the metrics layer
  - Token-budget adherence per tier
  - End-to-end grounding & latency per tier

Output: results/toon_consolidated_<ts>.csv
Columns: section, metric, scope, tier, value, unit, note
"""
from __future__ import annotations

import csv
import datetime as dt
from pathlib import Path

from app.core.config import cfg

RESULTS = Path(cfg.RESULTS_DIR)
TS = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = RESULTS / f"toon_consolidated_{TS}.csv"

rows = []  # each: dict(section, metric, scope, tier, value, unit, note)


def add(section, metric, scope, tier, value, unit="", note=""):
    rows.append(
        {
            "section": section,
            "metric": metric,
            "scope": scope,
            "tier": tier,
            "value": value,
            "unit": unit,
            "note": note,
        }
    )


# ── 1. Router accuracy (live benchmark) ───────────────────────────────────────
def bench_router():
    from app.retrieval.toon_router import classify, _classify_regex, detect_language
    from tests.test_rag_queries import TOON_TEST_QUERIES
    from eval_router_heldout import HELDOUT

    gold = {"tier_1_simple": 1, "tier_2_moderate": 2, "tier_3_complex": 3}
    testset = [(gold[g], TOON_TEST_QUERIES[g]["queries"]["questions"]) for g in gold]
    heldout = list(HELDOUT.items())

    def acc(fn, items):
        c = t = 0
        per = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
        for exp, qs in items:
            for q in qs:
                tier = fn(q)
                t += 1
                per[exp][1] += 1
                if tier == exp:
                    c += 1
                    per[exp][0] += 1
        return c, t, per

    regex_fn = lambda q: _classify_regex(q.strip(), detect_language(q)).tier
    hybrid_fn = lambda q: classify(q).tier

    for name, fn in [("regex_only", regex_fn), ("hybrid", hybrid_fn)]:
        for scope, items in [("test_set", testset), ("held_out", heldout)]:
            c, t, per = acc(fn, items)
            add("router", "accuracy", f"{name}/{scope}", "all", round(100 * c / t, 1), "%")
            for tr in (1, 2, 3):
                ok, n = per[tr]
                add("router", "recall", f"{name}/{scope}", tr, round(100 * ok / n, 1), "%")

    # learned classifier (held-out only — its honest generalization number)
    try:
        from app.retrieval.toon_classifier import EmbeddingRouter

        r = EmbeddingRouter()
        c, t, per = acc(lambda q: r.predict(q)[0], heldout)
        add("router", "accuracy", "ml_classifier/held_out", "all", round(100 * c / t, 1), "%")
        for tr in (1, 2, 3):
            ok, n = per[tr]
            add("router", "recall", "ml_classifier/held_out", tr, round(100 * ok / n, 1), "%")
    except Exception as e:
        add("router", "accuracy", "ml_classifier/held_out", "all", "NA", "%", f"unavailable: {e}")

    # historical baseline (documented finding before the fix)
    add("router", "accuracy", "original_baseline/test_set", "all", 34.0, "%",
        "pre-fix regex; Tier-1 lab queries never matched")


# ── 2. Retrieval IR metrics (from latest metrics-layer CSV) ───────────────────
def add_retrieval_metrics():
    files = sorted(RESULTS.glob("toon_retrieval_metrics_*.csv"))
    if not files:
        add("retrieval", "recall@5", "row_level_gt", "NA", "NA", "", "no metrics CSV found")
        return
    latest = files[-1]
    with open(latest) as f:
        for r in csv.DictReader(f):
            tier = r["tier"]
            for k in ("recall@1", "recall@5", "recall@10", "ndcg@5", "mrr", "precision@1"):
                add("retrieval", k, "row_level_gt(32B-judged)", tier, r[k], "",
                    f"n_scored={r['n_scored']} src={latest.name}")


# ── 3. Token-budget adherence (from latest budget summary) ────────────────────
def add_budget():
    files = sorted(RESULTS.glob("toon_retrieval_summary_*.csv"))
    if not files:
        return
    latest = files[-1]
    with open(latest) as f:
        for r in csv.DictReader(f):
            tier = r["tier"]
            add("budget", "avg_tokens", "retrieval", tier, r["avg_tokens"], "tokens",
                f"budget={r['budget']}")
            add("budget", "within_budget_rate", "retrieval", tier,
                round(100 * float(r["within_budget_rate"]), 1), "%",
                f"budget={r['budget']} avg={r['avg_tokens']}")
            add("budget", "retrieval_rate", "retrieval", tier,
                round(100 * float(r["retrieval_rate"]), 1), "%")


# ── 4. End-to-end grounding & latency (from latest endtoend CSV) ──────────────
def add_endtoend():
    files = sorted(RESULTS.glob("toon_endtoend_*.csv"))
    if not files:
        return
    latest = files[-1]
    agg = {}  # tier_used -> [grounding_sum, lat_sum, n]
    with open(latest) as f:
        for r in csv.DictReader(f):
            tu = r.get("tier_used", "?")
            g = float(r.get("grounding_score") or 0)
            lat = float(r.get("latency_seconds") or 0)
            a = agg.setdefault(tu, [0.0, 0.0, 0])
            a[0] += g
            a[1] += lat
            a[2] += 1
    for tu, (gs, ls, n) in sorted(agg.items()):
        add("endtoend", "grounding_score", "live_rag", tu, round(gs / n, 3), "", f"n={n}")
        add("endtoend", "latency", "live_rag", tu, round(ls / n, 2), "s", f"n={n}")


def main():
    bench_router()
    add_retrieval_metrics()
    add_budget()
    add_endtoend()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["section", "metric", "scope", "tier", "value", "unit", "note"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows -> {OUT}")
    # quick console preview of the headline router numbers
    for r in rows:
        if r["section"] == "router" and r["metric"] == "accuracy":
            print(f"  router accuracy [{r['scope']}] = {r['value']}%")


if __name__ == "__main__":
    main()
