"""
End-to-end answer-correctness evaluation (resumable).

Runs the FULL production pipeline (PatientRAGService.run) on every query, then
scores each generated answer with the INDEPENDENT judge LLM — for ALL tiers,
not just T3. This replaces the hardcoded grounding_score=0.9/0.85 that the
service returns for T1/T2.

Why previous runs died
----------------------
  * results were only saved at the very end -> a crash at 45/100 or 12/100 lost
    everything;
  * no resume -> every restart began from zero.

This version
------------
  * appends one JSON line per query to a checkpoint file and fsyncs immediately,
    so a crash loses at most the in-flight query;
  * resumes by skipping queries already present in the checkpoint;
  * wraps every query in try/except so a single failure cannot abort the run;
  * is serial by design (Groq free tier is rate-limited ~30 RPM — concurrency
    is what crashed the parallel attempt, not throughput).

Usage
-----
  CUDA_VISIBLE_DEVICES=7 python3 eval_endtoend.py            # run / resume
  CUDA_VISIBLE_DEVICES=7 python3 eval_endtoend.py --report   # summarize only

Outputs
-------
  results/endtoend_checkpoint.jsonl     append-only, one row per scored query
  results/endtoend_summary_{ts}.csv     overall + per-tier grounding / halluc
  results/plots/endtoend_{ts}.png       per-tier grounding + hallucination bars
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")
os.environ.setdefault("TOON_ROUTER_MODE", "hybrid")

QUERIES = Path("data/toon_multipatient_queries.json")
TIER_INT = {"tier_1_simple": 1, "tier_2_moderate": 2, "tier_3_complex": 3}

RESULTS = Path("results")
PLOTS = RESULTS / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)
CKPT = RESULTS / "endtoend_checkpoint.jsonl"


def _retrieve_context_texts(patient_id: int, query: str, tier: int) -> list[str]:
    """Reconstruct the context the pipeline used, as a list for the judge."""
    from app.retrieval.toon import search_bm25, search_hybrid, fetch_live_context
    if tier == 1:
        ctx = search_bm25(patient_id, query, top_k=5)
    elif tier == 2:
        ctx = search_hybrid(patient_id, query, top_k=10)
    else:
        ctx = fetch_live_context(patient_id)
    return [ctx] if isinstance(ctx, str) and ctx.strip() else ([] if not ctx else [str(ctx)])


def _load_done() -> set[str]:
    if not CKPT.exists():
        return set()
    done = set()
    for line in CKPT.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["query"])
        except Exception:
            continue
    return done


def _append(row: dict):
    with open(CKPT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run():
    queries = json.loads(QUERIES.read_text(encoding="utf-8"))
    done = _load_done()
    todo = [q for q in queries if q["query"] not in done]
    print(f"total={len(queries)}  done={len(done)}  todo={len(todo)}", flush=True)

    from app.retrieval.toon import load_patient_index, index_patient
    from app.retrieval.toon_service import PatientRAGService
    from app.safety.judge import judge_answer

    print("warming patient indexes …", flush=True)
    for pid in sorted({q["patient_id"] for q in queries}):
        vs, bm = load_patient_index(pid)
        if not vs and not bm:
            index_patient(pid)

    svc = PatientRAGService()
    t0 = time.time()
    for i, q in enumerate(todo, 1):
        query, pid = q["query"], q["patient_id"]
        try:
            res = svc.run(query, pid, role="patient")
            tier = res.tier_used or TIER_INT.get(q["tier"], 0)
            ctx_texts = _retrieve_context_texts(pid, query, tier)
            jr = judge_answer(query, res.answer, ctx_texts)
            row = {
                "query": query,
                "patient_id": pid,
                "expected_tier": TIER_INT.get(q["tier"], 0),
                "tier_used": tier,
                "category": q.get("category") or "uncategorized",
                "status": res.status,
                "answer": res.answer,
                "judge_grounded": bool(jr.grounded),
                "judge_grounding_score": float(jr.grounding_score),
                "judge_has_hallucination": bool(jr.has_hallucination),
                "judge_hallucination_risk": float(jr.hallucination_risk),
                "judge_confidence": float(jr.confidence),
                "judge_reasoning": jr.reasoning[:500],
                "n_context": len(ctx_texts),
            }
        except Exception as e:
            row = {
                "query": query, "patient_id": pid,
                "expected_tier": TIER_INT.get(q["tier"], 0), "tier_used": 0,
                "category": q.get("category") or "uncategorized",
                "status": f"error:{type(e).__name__}", "answer": "",
                "judge_grounded": False, "judge_grounding_score": 0.0,
                "judge_has_hallucination": False, "judge_hallucination_risk": 0.0,
                "judge_confidence": 0.0, "judge_reasoning": str(e)[:500],
                "n_context": 0,
            }
        _append(row)
        el = time.time() - t0
        print(f"  [{i}/{len(todo)}] tier={row['tier_used']} "
              f"ground={row['judge_grounding_score']:.2f} "
              f"halluc={row['judge_hallucination_risk']:.2f} "
              f"status={row['status']}  ({el:.0f}s, {el/i:.1f}s/q)", flush=True)

    print("\nall queries processed — generating summary", flush=True)
    report()


def report():
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.05)

    if not CKPT.exists():
        print("no checkpoint yet — run first", file=sys.stderr)
        return
    rows = [json.loads(l) for l in CKPT.read_text(encoding="utf-8").splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    ok = df[~df["status"].str.startswith("error")]
    print(f"\nscored rows: {len(ok)}/{len(df)} (errors: {len(df)-len(ok)})", flush=True)

    def block(sub):
        return {
            "n": len(sub),
            "grounded_rate":      round((sub["judge_grounded"]).mean(), 4),
            "mean_grounding":     round(sub["judge_grounding_score"].mean(), 4),
            "halluc_rate":        round((sub["judge_has_hallucination"]).mean(), 4),
            "mean_halluc_risk":   round(sub["judge_hallucination_risk"].mean(), 4),
            "mean_confidence":    round(sub["judge_confidence"].mean(), 4),
        }

    summary = [{"slice": "OVERALL", **block(ok)}]
    for tier in (1, 2, 3):
        sub = ok[ok["tier_used"] == tier]
        if len(sub):
            summary.append({"slice": f"T{tier}", **block(sub)})
    sdf = pd.DataFrame(summary)

    ts = time.strftime("%Y%m%d_%H%M%S")
    sout = RESULTS / f"endtoend_summary_{ts}.csv"
    sdf.to_csv(sout, index=False)

    print("\n" + "=" * 80)
    print("END-TO-END ANSWER CORRECTNESS  (independent judge, all tiers)")
    print("=" * 80)
    print(sdf.to_string(index=False))
    print(f"\nsaved: {sout}")

    tier_rows = [r for r in summary if r["slice"].startswith("T")]
    if tier_rows:
        pdf = pd.DataFrame(tier_rows).melt(
            id_vars="slice",
            value_vars=["grounded_rate", "mean_grounding", "halluc_rate", "mean_confidence"],
            var_name="metric", value_name="score")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=pdf, x="metric", y="score", hue="slice", ax=ax)
        ax.set_ylim(0, 1.0)
        ax.set_title("End-to-end answer quality by tier (independent judge)")
        ax.set_xlabel(""); plt.xticks(rotation=15)
        plt.tight_layout()
        pplot = PLOTS / f"endtoend_{ts}.png"
        plt.savefig(pplot, dpi=120); plt.close()
        print(f"saved: {pplot}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="summarize checkpoint only")
    args = ap.parse_args()
    if args.report:
        report()
    else:
        run()
