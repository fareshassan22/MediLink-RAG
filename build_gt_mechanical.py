#!/usr/bin/env python3
"""Mechanical ground-truth audit.

WHY: the v2 row-level ground truth was hand-authored by the same person who
wrote the queries. That is a validity hole — an author grading their own exam.
This script removes the subjectivity for the *objectively answerable* subset of
queries by deriving the correct relevant row from a deterministic rule applied
to the live patient data, then DIFFING that against the hand-authored labels.

It does NOT edit v1 or v2. It writes a NEW audit file. Where the mechanical rule
and the human label agree, the human label is validated. Where they disagree, a
candidate labeling bug is surfaced for review.

Objective rules implemented (only where intent is unambiguous via regex):
  - next_appointment : appointment row with the earliest scheduled_at >= now
  - last_appointment : appointment row with the latest   scheduled_at <= now

These mirror the same regexes the date-aware reranker uses, so the audit and the
runtime behaviour are derived from one shared, inspectable definition.

Usage:
    CUDA_VISIBLE_DEVICES=7 python3 build_gt_mechanical.py
Output:
    results/gt_mechanical_audit_<ts>.csv   (per-query agree/disagree)
    results/gt_mechanical_audit_<ts>.json  (machine-readable, with diffs)
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Reuse the SAME definitions the runtime reranker uses — single source of truth.
from app.retrieval.toon import (
    fetch_all_chunks,
    _parse_sched,
    _appt_status,
    _NEXT_INTENT,
    _LAST_INTENT,
    _APPT_HINT,
)

GT_PATH = Path("data/toon_rowlevel_ground_truth_multipatient_v2.json")
OUT_DIR = Path("results")
NOW = datetime.now()


def _appt_rows(patient_id: int) -> List[Dict]:
    """All appointment rows for a patient as {doc_id, sched, status}."""
    rows = []
    for ch in fetch_all_chunks(patient_id):
        if ch.get("metadata", {}).get("table") != "appointments":
            continue
        rows.append(
            {
                "doc_id": ch["metadata"]["doc_id"],
                "sched": _parse_sched(ch.get("text", "")),
                "status": _appt_status(ch.get("text", "")),
            }
        )
    return rows


def _mechanical_appt(query: str, rows: List[Dict]) -> Optional[str]:
    """Return the doc_id the objective rule selects, or None if no rule applies.

    Rule fires only when the query unambiguously asks for the next OR the last
    appointment (exactly one intent) and mentions an appointment. Same gating and
    same definitions as the runtime reranker, so the audit reflects shipped
    behaviour:
      next = earliest appointment still in the future.
      last = most recent *completed* visit (fallback: most recent past slot).
    """
    want_next = bool(_NEXT_INTENT.search(query))
    want_last = bool(_LAST_INTENT.search(query))
    if want_next == want_last:          # neither, or ambiguous both
        return None
    if not _APPT_HINT.search(query):
        return None
    dated = [r for r in rows if r["sched"] is not None]
    if not dated:
        return None
    if want_next:
        future = [r for r in dated if r["sched"] >= NOW]
        pick = min(future, key=lambda r: r["sched"]) if future else None
    else:  # want_last — most recent visit that actually happened
        completed = [r for r in dated if r["status"] == "completed" and r["sched"] <= NOW]
        past = [r for r in dated if r["sched"] <= NOW]
        if completed:
            pick = max(completed, key=lambda r: r["sched"])
        elif past:
            pick = max(past, key=lambda r: r["sched"])
        else:
            pick = None
    return pick["doc_id"] if pick else None


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    gt = json.loads(GT_PATH.read_text(encoding="utf-8"))

    # Cache appointment rows per patient (network calls are the slow part).
    appt_cache: Dict[int, List[Dict]] = {}

    audited: List[Dict] = []
    n_rule = n_agree = n_disagree = 0

    for entry in gt:
        q = entry["query"]
        pid = entry["patient_id"]
        if pid not in appt_cache:
            try:
                appt_cache[pid] = _appt_rows(pid)
            except Exception as e:  # network / data issue for this patient
                appt_cache[pid] = []
                print(f"  ! patient {pid}: fetch failed ({type(e).__name__})")

        mech = _mechanical_appt(q, appt_cache[pid])
        if mech is None:
            continue  # no objective rule applies — out of audit scope

        n_rule += 1
        human_appts = [r for r in entry.get("relevant_ids", []) if r.startswith("appointments_")]
        agree = mech in human_appts
        n_agree += int(agree)
        n_disagree += int(not agree)
        audited.append(
            {
                "query": q,
                "patient_id": pid,
                "rule": "next_appointment" if _NEXT_INTENT.search(q) else "last_appointment",
                "mechanical_id": mech,
                "human_appt_ids": human_appts,
                "agree": agree,
            }
        )
        flag = "OK " if agree else "XX "
        print(f"  {flag}p{pid} {q[:34]:34s} mech={mech} human={human_appts}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    # CSV
    import csv

    csv_path = OUT_DIR / f"gt_mechanical_audit_{ts}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query", "patient_id", "rule", "mechanical_id", "human_appt_ids", "agree"])
        for a in audited:
            w.writerow([a["query"], a["patient_id"], a["rule"], a["mechanical_id"],
                        ";".join(a["human_appt_ids"]), a["agree"]])
    # JSON
    json_path = OUT_DIR / f"gt_mechanical_audit_{ts}.json"
    summary = {
        "generated_at": ts,
        "gt_file": str(GT_PATH),
        "queries_total": len(gt),
        "queries_with_objective_rule": n_rule,
        "agree": n_agree,
        "disagree": n_disagree,
        "agreement_rate": round(n_agree / n_rule, 3) if n_rule else None,
        "disagreements": [a for a in audited if not a["agree"]],
        "audited": audited,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"queries with an objective rule : {n_rule} / {len(gt)}")
    print(f"agree with human label         : {n_agree}")
    print(f"disagree (candidate bugs)      : {n_disagree}")
    if n_rule:
        print(f"agreement rate                 : {n_agree / n_rule:.1%}")
    print(f"\nwrote {csv_path}")
    print(f"wrote {json_path}")
    if n_disagree:
        print("\nDisagreements to review (NOT auto-applied):")
        for a in summary["disagreements"]:
            print(f"  p{a['patient_id']} [{a['rule']}] {a['query']}")
            print(f"     mechanical={a['mechanical_id']}  human={a['human_appt_ids']}")


if __name__ == "__main__":
    main()
