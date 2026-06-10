"""
GT validity — real-query distribution gap analyzer.

Honest finding this script quantifies: the production logs show users ask
GENERAL medical-knowledge questions ("symptoms of diabetes?"), while the
evaluation ground truth tests PATIENT-RECORD retrieval ("patient X's
temperature"). This script measures that mismatch so it can be reported.

It does NOT touch the GPU or Groq (router forced to regex mode), so it is safe
to run while the end-to-end eval is in progress.

Inputs
------
  logs/medilink_*.jsonl                          real user queries (query field)
  data/toon_multipatient_queries.json            eval GT queries

Outputs
-------
  results/gt_distribution_gap_{ts}.csv           per-real-query: tier, intent
  prints a side-by-side summary of the two distributions
"""
from __future__ import annotations
import glob, json, os, re, time
from collections import Counter
from pathlib import Path

os.environ.setdefault("TOON_ROUTER_MODE", "regex")   # no ML model -> no GPU/Groq

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

EVAL_QUERIES = Path("data/toon_multipatient_queries.json")
TIER_INT = {"tier_1_simple": 1, "tier_2_moderate": 2, "tier_3_complex": 3}

# Patient-record intent: query asks about a specific patient's stored data.
# General-knowledge intent: query asks about a disease/symptom in the abstract.
_PATIENT_MARKERS = re.compile(
    r"(my|patient|المريض|مريض|عندي|لدي|نتيجة|نتائجي|موعد|مواعيد|"
    r"وصفت|دوائي|أدويتي|تحليلي|تحاليلي|ضغطي|حرارتي|سجل|تقرير|"
    r"appointment|prescription|my result|my lab|my vitals|my record)",
    re.IGNORECASE,
)
_GENERAL_MARKERS = re.compile(
    r"(أسباب|أعراض|ما هي|ما هو|علاج|الفرق بين|symptoms|causes|what (is|are)|"
    r"treatment|relationship between|how (to|do)|why does)",
    re.IGNORECASE,
)


def classify_intent(q: str) -> str:
    has_patient = bool(_PATIENT_MARKERS.search(q))
    has_general = bool(_GENERAL_MARKERS.search(q))
    if has_patient and not has_general:
        return "patient_record"
    if has_general and not has_patient:
        return "general_knowledge"
    if has_patient and has_general:
        return "mixed"
    return "ambiguous"


def load_real_queries() -> list[str]:
    seen, out = set(), []
    for fp in sorted(glob.glob("logs/medilink_*.jsonl")):
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line).get("query", "").strip()
            except Exception:
                continue
            if q and q not in seen:
                seen.add(q)
                out.append(q)
    return out


def main():
    real = load_real_queries()
    eval_qs = json.loads(EVAL_QUERIES.read_text(encoding="utf-8"))
    print(f"real unique queries: {len(real)}   eval GT queries: {len(eval_qs)}", flush=True)

    from app.retrieval.toon_router import classify as route

    # Real queries: intent + router tier.
    rows = []
    real_intent = Counter()
    real_tier = Counter()
    for q in real:
        intent = classify_intent(q)
        try:
            tier = route(q, 0).tier
        except Exception:
            tier = 0
        real_intent[intent] += 1
        real_tier[f"T{tier}"] += 1
        rows.append({"query": q, "intent": intent, "router_tier": tier})

    # Eval GT queries: intent + declared tier.
    eval_intent = Counter()
    eval_tier = Counter()
    for e in eval_qs:
        eval_intent[classify_intent(e["query"])] += 1
        eval_tier[f"T{TIER_INT.get(e['tier'], 0)}"] += 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS / f"gt_distribution_gap_{ts}.csv"
    import csv
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query", "intent", "router_tier"])
        w.writeheader()
        w.writerows(rows)

    def pct(counter, total):
        return {k: f"{v} ({100*v/max(1,total):.0f}%)" for k, v in counter.items()}

    print("\n" + "=" * 72)
    print("GT DISTRIBUTION GAP — real user queries vs evaluation GT")
    print("=" * 72)
    print(f"\nINTENT  (what the query is actually asking for)")
    print(f"  REAL  (n={len(real)}):  {dict(pct(real_intent, len(real)))}")
    print(f"  EVAL  (n={len(eval_qs)}): {dict(pct(eval_intent, len(eval_qs)))}")
    print(f"\nROUTER/DECLARED TIER")
    print(f"  REAL  routed:   {dict(pct(real_tier, len(real)))}")
    print(f"  EVAL  declared: {dict(pct(eval_tier, len(eval_qs)))}")

    gen = real_intent.get("general_knowledge", 0)
    pat = eval_intent.get("patient_record", 0) + eval_intent.get("mixed", 0)
    print("\nHONEST READING")
    print(f"  {100*gen/max(1,len(real)):.0f}% of REAL queries are general medical knowledge.")
    print(f"  {100*pat/max(1,len(eval_qs)):.0f}% of EVAL GT queries are patient-record lookups.")
    print("  => The retrieval GT measures a task users rarely perform. Report this gap;")
    print("     scope retrieval claims to 'patient-record retrieval', not 'user queries'.")
    print(f"\nsaved: {out}")


if __name__ == "__main__":
    main()
