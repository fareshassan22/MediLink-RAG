"""
Honest end-to-end evaluation of the TOON pipeline.

Why this script exists (and why we can't trust the service's own numbers):
- app/retrieval/toon_service.py returns HARDCODED grounding_score = 0.9 for T1
  and 0.85 for T2. Those are constants, not measurements. Only T3 actually
  calls the judge.
- To honestly answer "do the final Arabic answers match the patient data?"
  we must judge EVERY answer with the same real judge, regardless of tier.

What we measure:
1. Pipeline runs                 — answer text + tier_used + latency
2. LLM-judge grounding           — same llama-3.1-8b judge for all tiers
                                   (with explicit caveat: judge != truth)
3. Deterministic signals         — refusal rate, empty-answer rate, length,
                                   service-says-error rate. These do NOT need
                                   an LLM and cannot be fudged.
4. Per-category & per-tier rollups
5. Per-query CSV for full audit

Output:
  results/endtoend_honest_{ts}.csv     (one row per query, all fields)
  results/endtoend_summary_{ts}.json   (aggregate stats)
  results/plots/endtoend_*.png         (charts)
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
os.environ.setdefault("TOON_ROUTER_MODE", "hybrid")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from app.core.config import cfg

RESULTS = Path(cfg.RESULTS_DIR)
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)

QUERIES = Path("data/toon_multipatient_queries.json")

TIER_INT = {"tier_1_simple": 1, "tier_2_moderate": 2, "tier_3_complex": 3}

# Refusal phrases — both Arabic and English. Used for deterministic
# "did the system refuse to answer?" metric. These are exact substrings.
REFUSAL_PHRASES = [
    "لا تتوفر",
    "لا توجد بيانات",
    "لا أستطيع",
    "لا يمكنني",
    "غير متوفر",
    "no information",
    "cannot provide",
    "i don't have",
    "unable to",
    "no data",
]


def is_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p.lower() in a for p in REFUSAL_PHRASES)


def main():
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    print(f"loaded {len(queries)} queries")

    # Lazy import — these pull Supabase, embedder, LLM client
    from app.retrieval.toon_service import patient_rag_service
    from app.retrieval.toon import (
        load_patient_index, index_patient,
        search_bm25, search_hybrid, fetch_live_context,
    )
    from app.safety.judge import judge_answer

    # Warm patient indexes once
    for pid in sorted({q["patient_id"] for q in queries}):
        vs, bm = load_patient_index(pid)
        if not vs and not bm:
            index_patient(pid)

    def capture_context(pid: int, query: str, tier_used: int) -> str:
        """Re-fetch the context the pipeline actually used, so we can judge."""
        try:
            if tier_used == 1:
                return search_bm25(pid, query, top_k=5) or ""
            if tier_used == 2:
                return search_hybrid(pid, query, top_k=10) or ""
            if tier_used == 3:
                return fetch_live_context(pid) or ""
        except Exception:
            return ""
        return ""

    rows = []
    t_start = time.time()
    for i, q in enumerate(queries, 1):
        query = q["query"]
        pid = q["patient_id"]
        exp_tier = TIER_INT[q["tier"]]
        category = q.get("category") or "uncategorized"

        # --- 1. Run pipeline -------------------------------------------------
        r0 = time.time()
        try:
            res = patient_rag_service.run(query=query, patient_id=pid)
            answer = res.answer or ""
            tier_used = res.tier_used
            status = res.status
            service_grounding = res.grounding_score  # hardcoded for T1/T2!
            err = ""
        except Exception as e:
            answer = ""
            tier_used = -1
            status = f"error:{type(e).__name__}"
            service_grounding = 0.0
            err = str(e)
        latency_pipeline = round(time.time() - r0, 3)

        # --- 2. Deterministic signals (cannot be faked) ---------------------
        empty_answer = (len(answer.strip()) == 0)
        refused = (not empty_answer) and is_refusal(answer)
        n_chars = len(answer)

        # --- 3. Honest LLM judge --------------------------------------------
        # Only judge when there is an answer AND a context to ground in.
        # Use the SAME judge for all tiers.
        if answer and not refused and tier_used > 0:
            ctx = capture_context(pid, query, tier_used)
            j0 = time.time()
            try:
                jr = judge_answer(query=query, answer=answer, context_texts=[ctx[:3000]])
                judge_grounded = jr.grounded
                judge_grounding = jr.grounding_score
                judge_halluc = jr.hallucination_risk
                judge_conf = jr.confidence
                judge_reason = jr.reasoning[:200]
            except Exception as e:
                judge_grounded = None
                judge_grounding = None
                judge_halluc = None
                judge_conf = None
                judge_reason = f"judge_error:{type(e).__name__}"
            latency_judge = round(time.time() - j0, 3)
        else:
            judge_grounded = None
            judge_grounding = None
            judge_halluc = None
            judge_conf = None
            judge_reason = "skipped (empty or refused)"
            latency_judge = 0.0

        rows.append({
            "i": i,
            "query": query,
            "patient_id": pid,
            "category": category,
            "expected_tier": exp_tier,
            "tier_used": tier_used,
            "status": status,
            "answer": answer,
            "n_chars": n_chars,
            "empty_answer": int(empty_answer),
            "refused": int(refused),
            "service_grounding_hardcoded": service_grounding,
            "judge_grounded": judge_grounded,
            "judge_grounding": judge_grounding,
            "judge_halluc_risk": judge_halluc,
            "judge_confidence": judge_conf,
            "judge_reason": judge_reason,
            "latency_pipeline_s": latency_pipeline,
            "latency_judge_s": latency_judge,
            "error": err,
        })

        # heartbeat
        if i % 5 == 0 or i == len(queries):
            elapsed = time.time() - t_start
            print(f"  [{i:3d}/{len(queries)}] T{tier_used} {latency_pipeline:5.2f}s "
                  f"+ judge {latency_judge:5.2f}s | refused={refused} "
                  f"jg={judge_grounding}  (elapsed {elapsed:5.0f}s)")

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_q = RESULTS / f"endtoend_honest_{ts}.csv"
    df.to_csv(per_q, index=False)
    print(f"\nsaved per-query: {per_q}")

    # ─── Aggregate stats ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("HONEST END-TO-END SUMMARY")
    print("=" * 72)

    ok = df[df["tier_used"] > 0]
    n = len(df)
    n_ok = len(ok)
    n_empty = int(df["empty_answer"].sum())
    n_refused = int(df["refused"].sum())
    n_errors = int((df["tier_used"] < 0).sum())

    print(f"\nDeterministic (not LLM-judged):")
    print(f"  queries:                 {n}")
    print(f"  pipeline succeeded:      {n_ok}/{n}  = {n_ok/n:.0%}")
    print(f"  pipeline errors:         {n_errors}")
    print(f"  empty answers:           {n_empty}/{n}  = {n_empty/n:.0%}")
    print(f"  explicit refusals:       {n_refused}/{n} = {n_refused/n:.0%}")
    print(f"  avg latency (pipeline):  {ok['latency_pipeline_s'].mean():.2f}s")
    print(f"  median latency:          {ok['latency_pipeline_s'].median():.2f}s")
    print(f"  p95 latency:             {ok['latency_pipeline_s'].quantile(0.95):.2f}s")

    judged = df.dropna(subset=["judge_grounding"])
    print(f"\nLLM judge (n={len(judged)}, llama-3.1-8b-instant; caveat: judge != truth):")
    if len(judged):
        print(f"  mean grounding:          {judged['judge_grounding'].mean():.3f}")
        print(f"  median grounding:        {judged['judge_grounding'].median():.3f}")
        print(f"  grounded (>=0.7):        {(judged['judge_grounding']>=0.7).mean():.0%}")
        print(f"  grounded (>=0.5):        {(judged['judge_grounding']>=0.5).mean():.0%}")
        print(f"  mean hallucination risk: {judged['judge_halluc_risk'].mean():.3f}")
        print(f"  high halluc risk (>0.5): {(judged['judge_halluc_risk']>0.5).mean():.0%}")

    print(f"\nPer-tier (judged answers only):")
    print(f"  {'tier':<6}{'n':<5}{'refused%':<10}{'mean_grounding':<16}{'>=0.7%':<10}{'avg_latency_s'}")
    for t in (1, 2, 3):
        sub = df[df["tier_used"] == t]
        if not len(sub):
            continue
        sub_j = sub.dropna(subset=["judge_grounding"])
        refused_pct = sub["refused"].mean()
        mean_g = sub_j["judge_grounding"].mean() if len(sub_j) else float("nan")
        g70 = (sub_j["judge_grounding"] >= 0.7).mean() if len(sub_j) else float("nan")
        lat = sub["latency_pipeline_s"].mean()
        print(f"  T{t:<5}{len(sub):<5}{refused_pct:.0%}      "
              f"{mean_g:<16.3f}{g70:.0%}      {lat:.2f}")

    print(f"\nPer-category (judged answers only):")
    print(f"  {'category':<14}{'n':<5}{'refused%':<10}{'mean_grounding':<16}{'>=0.7%'}")
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        sub_j = sub.dropna(subset=["judge_grounding"])
        refused_pct = sub["refused"].mean()
        mean_g = sub_j["judge_grounding"].mean() if len(sub_j) else float("nan")
        g70 = (sub_j["judge_grounding"] >= 0.7).mean() if len(sub_j) else float("nan")
        print(f"  {cat:<14}{len(sub):<5}{refused_pct:.0%}      "
              f"{mean_g:<16.3f}{g70:.0%}")

    # Honesty: how often does the service's hardcoded score disagree with the judge?
    j = df.dropna(subset=["judge_grounding"])
    if len(j):
        gap = (j["service_grounding_hardcoded"] - j["judge_grounding"]).abs()
        print(f"\n|service hardcoded - real judge| (the 'faking' delta):")
        print(f"  mean abs gap: {gap.mean():.3f}")
        print(f"  median:       {gap.median():.3f}")
        print(f"  > 0.2 gap:    {(gap > 0.2).mean():.0%} of queries")

    # ─── Plots ──────────────────────────────────────────────────────────────
    if len(judged):
        fig, ax = plt.subplots(figsize=(8, 5))
        for t in (1, 2, 3):
            sub = judged[judged["tier_used"] == t]
            if len(sub):
                ax.hist(sub["judge_grounding"], bins=20, alpha=0.5, label=f"T{t} (n={len(sub)})")
        ax.set_xlabel("LLM-judge grounding score")
        ax.set_ylabel("queries")
        ax.set_title("End-to-end grounding (llama-3.1-8b judge)")
        ax.legend()
        plt.tight_layout()
        p = PLOTS / f"endtoend_grounding_hist_{ts}.png"
        plt.savefig(p, dpi=120); plt.close()
        print(f"\nsaved {p}")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=ok, x="tier_used", y="latency_pipeline_s", ax=ax)
        ax.set_title("End-to-end latency by tier (pipeline only, excludes judge)")
        ax.set_xlabel("Tier")
        ax.set_ylabel("Latency (s)")
        plt.tight_layout()
        p = PLOTS / f"endtoend_latency_{ts}.png"
        plt.savefig(p, dpi=120); plt.close()
        print(f"saved {p}")

    # summary json
    summary = {
        "timestamp": ts,
        "n_queries": n,
        "n_pipeline_ok": n_ok,
        "n_empty": n_empty,
        "n_refused": n_refused,
        "n_errors": n_errors,
        "latency_pipeline_s": {
            "mean": round(ok["latency_pipeline_s"].mean(), 3),
            "median": round(ok["latency_pipeline_s"].median(), 3),
            "p95": round(ok["latency_pipeline_s"].quantile(0.95), 3),
        },
        "judge": {
            "n_judged": int(len(judged)),
            "mean_grounding": round(float(judged["judge_grounding"].mean()), 4) if len(judged) else None,
            "grounded_rate_0_7": round(float((judged["judge_grounding"] >= 0.7).mean()), 4) if len(judged) else None,
            "mean_halluc_risk": round(float(judged["judge_halluc_risk"].mean()), 4) if len(judged) else None,
        },
        "per_tier": {
            f"T{t}": {
                "n": int(len(df[df["tier_used"] == t])),
                "refused_rate": round(float(df[df["tier_used"] == t]["refused"].mean()), 4) if len(df[df["tier_used"] == t]) else None,
                "mean_grounding": round(float(df[(df["tier_used"] == t) & df["judge_grounding"].notna()]["judge_grounding"].mean()), 4)
                                  if len(df[(df["tier_used"] == t) & df["judge_grounding"].notna()]) else None,
                "avg_latency_s": round(float(df[df["tier_used"] == t]["latency_pipeline_s"].mean()), 3)
                                  if len(df[df["tier_used"] == t]) else None,
            } for t in (1, 2, 3) if len(df[df["tier_used"] == t])
        },
    }
    sjson = RESULTS / f"endtoend_summary_{ts}.json"
    sjson.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved summary: {sjson}")

    print("\nDONE.")


if __name__ == "__main__":
    main()
