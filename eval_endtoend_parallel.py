"""
Honest end-to-end evaluation with PARALLEL Groq calls.

Bottleneck of the serial version was Groq HTTP latency (~27s/query waiting on
remote LLMs). GPU was idle. This version runs the pipeline concurrently with
a small worker pool so we hit Groq in parallel without exceeding rate limits.

Same measurements as eval_endtoend_honest.py:
  1. Pipeline runs (answer / tier_used / latency / status)
  2. Independent LLM-judge grounding (overrides hardcoded T1=0.9 / T2=0.85)
  3. Deterministic signals (refusal rate, empty-answer rate, length)

Output:
  results/endtoend_honest_{ts}.csv
  results/endtoend_summary_{ts}.json
  results/plots/endtoend_*.png

Heartbeat is flushed line-by-line so we can watch progress live.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

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

# Parallelism.  Groq free tier is ~30 RPM on llama-3.1-8b — 8 workers gives us
# safe headroom (each query makes ~2 Groq calls = ~16 RPM peak).
WORKERS = int(os.environ.get("ENDTOEND_WORKERS", "8"))

REFUSAL_PHRASES = [
    "لا تتوفر", "لا توجد بيانات", "لا أستطيع", "لا يمكنني", "غير متوفر",
    "no information", "cannot provide", "i don't have", "unable to", "no data",
]


def is_refusal(answer: str) -> bool:
    a = (answer or "").lower()
    return any(p.lower() in a for p in REFUSAL_PHRASES)


def main():
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    print(f"loaded {len(queries)} queries; running with {WORKERS} workers", flush=True)

    from app.retrieval.toon_service import patient_rag_service
    from app.retrieval.toon import (
        load_patient_index, index_patient,
        search_bm25, search_hybrid, fetch_live_context,
    )
    from app.safety.judge import judge_answer

    # Warm patient indexes serially (avoids race in _INDEX_CACHE)
    print("warming patient indexes …", flush=True)
    for pid in sorted({q["patient_id"] for q in queries}):
        vs, bm = load_patient_index(pid)
        if not vs and not bm:
            index_patient(pid)
    print("indexes ready", flush=True)

    def run_one(idx: int, q: dict) -> dict:
        query = q["query"]
        pid = q["patient_id"]
        exp_tier = TIER_INT[q["tier"]]
        category = q.get("category") or "uncategorized"

        # 1) Pipeline
        r0 = time.time()
        try:
            res = patient_rag_service.run(query=query, patient_id=pid)
            answer = res.answer or ""
            tier_used = res.tier_used
            status = res.status
            service_g = res.grounding_score
            err = ""
        except Exception as e:
            answer = ""
            tier_used = -1
            status = f"error:{type(e).__name__}"
            service_g = 0.0
            err = str(e)
        lat_pipe = round(time.time() - r0, 3)

        # 2) Deterministic
        empty = (len(answer.strip()) == 0)
        refused = (not empty) and is_refusal(answer)

        # 3) LLM judge (only if there's something to judge)
        if answer and not refused and tier_used > 0:
            try:
                if tier_used == 1:
                    ctx = search_bm25(pid, query, top_k=5) or ""
                elif tier_used == 2:
                    ctx = search_hybrid(pid, query, top_k=10) or ""
                else:
                    ctx = fetch_live_context(pid) or ""
            except Exception:
                ctx = ""
            j0 = time.time()
            try:
                jr = judge_answer(query=query, answer=answer, context_texts=[ctx[:3000]])
                jg, jh, jc, jr_text = jr.grounding_score, jr.hallucination_risk, jr.confidence, jr.reasoning[:200]
                jgr = jr.grounded
            except Exception as e:
                jg = jh = jc = None
                jgr = None
                jr_text = f"judge_error:{type(e).__name__}"
            lat_judge = round(time.time() - j0, 3)
        else:
            jg = jh = jc = jgr = None
            jr_text = "skipped"
            lat_judge = 0.0

        return {
            "i": idx, "query": query, "patient_id": pid, "category": category,
            "expected_tier": exp_tier, "tier_used": tier_used, "status": status,
            "answer": answer, "n_chars": len(answer),
            "empty_answer": int(empty), "refused": int(refused),
            "service_grounding_hardcoded": service_g,
            "judge_grounded": jgr, "judge_grounding": jg,
            "judge_halluc_risk": jh, "judge_confidence": jc,
            "judge_reason": jr_text,
            "latency_pipeline_s": lat_pipe, "latency_judge_s": lat_judge,
            "error": err,
        }

    print_lock = Lock()
    rows = []
    t_start = time.time()
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(run_one, i + 1, q): i for i, q in enumerate(queries)}
        for fut in as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                # Should not happen — run_one already catches — but be safe.
                row = {"i": -1, "error": f"future:{type(e).__name__}:{e}",
                       "tier_used": -1, "judge_grounding": None}
            rows.append(row)
            done += 1
            with print_lock:
                elapsed = time.time() - t_start
                rate = done / max(elapsed, 1)
                eta = (len(queries) - done) / max(rate, 1e-6)
                jg = row.get("judge_grounding")
                jg_s = f"{jg:.2f}" if isinstance(jg, (int, float)) else "—"
                print(f"  [{done:3d}/{len(queries)}] T{row.get('tier_used','?')} "
                      f"pipe={row.get('latency_pipeline_s', 0):5.1f}s "
                      f"judge={row.get('latency_judge_s', 0):5.1f}s "
                      f"jg={jg_s}  refused={bool(row.get('refused'))}  "
                      f"| elapsed {elapsed:5.0f}s  ETA {eta:5.0f}s",
                      flush=True)

    # Sort by original order
    rows.sort(key=lambda r: r.get("i", 0))
    df = pd.DataFrame(rows)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    per_q = RESULTS / f"endtoend_honest_{ts}.csv"
    df.to_csv(per_q, index=False)
    print(f"\nsaved per-query: {per_q}", flush=True)

    # ─── Aggregate ────────────────────────────────────────────────────────
    print("\n" + "=" * 72, flush=True)
    print("HONEST END-TO-END SUMMARY", flush=True)
    print("=" * 72, flush=True)

    n = len(df)
    ok = df[df["tier_used"] > 0]
    judged = df.dropna(subset=["judge_grounding"])

    print(f"\nDeterministic (no LLM bias):")
    print(f"  queries:                 {n}")
    print(f"  pipeline succeeded:      {len(ok)}/{n} = {len(ok)/n:.0%}")
    print(f"  pipeline errors:         {int((df['tier_used']<0).sum())}")
    print(f"  empty answers:           {int(df['empty_answer'].sum())}/{n}")
    print(f"  explicit refusals:       {int(df['refused'].sum())}/{n} = {df['refused'].mean():.0%}")
    print(f"  avg pipeline latency:    {ok['latency_pipeline_s'].mean():.2f}s")
    print(f"  median:                  {ok['latency_pipeline_s'].median():.2f}s")
    print(f"  p95:                     {ok['latency_pipeline_s'].quantile(0.95):.2f}s")

    if len(judged):
        print(f"\nLLM judge (n={len(judged)}, llama-3.1-8b; caveat: judge != truth):")
        print(f"  mean grounding:          {judged['judge_grounding'].mean():.3f}")
        print(f"  median grounding:        {judged['judge_grounding'].median():.3f}")
        print(f"  grounded (>=0.7):        {(judged['judge_grounding']>=0.7).mean():.0%}")
        print(f"  grounded (>=0.5):        {(judged['judge_grounding']>=0.5).mean():.0%}")
        print(f"  mean hallucination risk: {judged['judge_halluc_risk'].mean():.3f}")
        print(f"  high halluc risk (>0.5): {(judged['judge_halluc_risk']>0.5).mean():.0%}")

    print(f"\nPer tier:")
    print(f"  {'tier':<6}{'n':<5}{'refused%':<10}{'mean_grounding':<16}{'>=0.7%':<9}{'avg_lat'}")
    for t in (1, 2, 3):
        sub = df[df["tier_used"] == t]
        if not len(sub): continue
        sj = sub.dropna(subset=["judge_grounding"])
        mg = sj["judge_grounding"].mean() if len(sj) else float("nan")
        g70 = (sj["judge_grounding"] >= 0.7).mean() if len(sj) else float("nan")
        print(f"  T{t:<5}{len(sub):<5}{sub['refused'].mean():.0%}      "
              f"{mg:<16.3f}{g70:<9.0%}{sub['latency_pipeline_s'].mean():.2f}s")

    print(f"\nPer category:")
    print(f"  {'category':<14}{'n':<5}{'refused%':<10}{'mean_grounding':<16}{'>=0.7%'}")
    for cat in sorted(df["category"].unique()):
        sub = df[df["category"] == cat]
        sj = sub.dropna(subset=["judge_grounding"])
        mg = sj["judge_grounding"].mean() if len(sj) else float("nan")
        g70 = (sj["judge_grounding"] >= 0.7).mean() if len(sj) else float("nan")
        print(f"  {cat:<14}{len(sub):<5}{sub['refused'].mean():.0%}      "
              f"{mg:<16.3f}{g70:.0%}")

    if len(judged):
        gap = (judged["service_grounding_hardcoded"] - judged["judge_grounding"]).abs()
        print(f"\n|service_hardcoded - real_judge|  (the 'faking' gap T1/T2 hides):")
        print(f"  mean abs gap: {gap.mean():.3f}")
        print(f"  >0.2 gap on:  {(gap>0.2).mean():.0%} of queries")

    # Plots
    if len(judged):
        fig, ax = plt.subplots(figsize=(8, 5))
        for t in (1, 2, 3):
            sub = judged[judged["tier_used"] == t]
            if len(sub):
                ax.hist(sub["judge_grounding"], bins=20, alpha=0.5, label=f"T{t} (n={len(sub)})")
        ax.set_xlabel("LLM-judge grounding"); ax.set_ylabel("queries")
        ax.set_title("End-to-end grounding (llama-3.1-8b judge, all tiers)")
        ax.legend(); plt.tight_layout()
        p = PLOTS / f"endtoend_grounding_hist_{ts}.png"
        plt.savefig(p, dpi=120); plt.close()
        print(f"\nsaved {p}")

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=ok, x="tier_used", y="latency_pipeline_s", ax=ax)
        ax.set_title("End-to-end latency by tier")
        ax.set_xlabel("Tier"); ax.set_ylabel("Latency (s)")
        plt.tight_layout()
        p = PLOTS / f"endtoend_latency_{ts}.png"
        plt.savefig(p, dpi=120); plt.close()
        print(f"saved {p}")

    summary = {
        "timestamp": ts, "workers": WORKERS, "n_queries": n,
        "n_pipeline_ok": int(len(ok)),
        "n_empty": int(df["empty_answer"].sum()),
        "n_refused": int(df["refused"].sum()),
        "n_errors": int((df["tier_used"] < 0).sum()),
        "wall_clock_s": round(time.time() - t_start, 1),
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
    }
    sjson = RESULTS / f"endtoend_summary_{ts}.json"
    sjson.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved summary: {sjson}")
    print(f"\nwall-clock: {summary['wall_clock_s']}s "
          f"(serial would have been ~{int(28*n)}s)")


if __name__ == "__main__":
    main()
