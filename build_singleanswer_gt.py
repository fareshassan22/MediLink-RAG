"""
Build a v2 row-level GT that fixes the "single-answer over-labeling" bug.

The v1 GT marks ALL appointments / vitals / labs as relevant for queries with
explicit single-answer intent like "when is my **next** appointment?". That
structurally caps precision/recall@1.

This script ONLY modifies queries with unambiguous temporal-singular intent:
    next | last | latest | current | recent | upcoming | most recent
    القادم | القادمة | التالي | التالية | الأخير | الأخيرة | الحالي | الحالية

For each such query, we pick the row deterministically from the indexed chunk
text (which is structured: `scheduled_at: <ISO> | status: <s> | ...`) and
replace `relevant_ids` with the single correct row(s).

All other queries (lists, descriptions, comparisons) keep their v1 GT unchanged.
Every modified entry carries `_v2_modified: true` and `_v2_rule: "<name>"`
so the change is fully auditable.

Output: data/toon_rowlevel_ground_truth_multipatient_v2.json
v1 file is NEVER overwritten.
"""
from __future__ import annotations
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

from app.retrieval.toon import load_patient_index, index_patient

V1_PATH = Path("data/toon_rowlevel_ground_truth_multipatient.json")
V2_PATH = Path("data/toon_rowlevel_ground_truth_multipatient_v2.json")

# Intent triggers — must be unambiguous "single answer" cues.
NEXT_RX = re.compile(
    r"\bnext\b|\bupcoming\b|القادم|القادمة|التالي|التالية", re.I
)
LAST_RX = re.compile(
    r"\blast\b|\blatest\b|\bmost\s+recent\b|\brecent\b|"
    r"الأخير|الأخيرة|آخر|الأحدث",
    re.I,
)
CURRENT_RX = re.compile(r"\bcurrent\b|\bnow\b|الحالي|الحالية|حالياً", re.I)


def _parse_kv(text: str) -> Dict[str, str]:
    """Parse `key: value | key: value` chunks into a dict."""
    body = text.split("]", 1)[-1]  # strip [source: x] prefix if present
    out: Dict[str, str] = {}
    for part in body.split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip().lower()] = v.strip()
    return out


def _doc_iter(vs):
    """Yield (doc_id, text, parsed_kv, source) for every chunk."""
    for d in vs.documents:
        text = getattr(d, "text", None) or (d.get("text", "") if isinstance(d, dict) else "")
        md = getattr(d, "metadata", None) or (d.get("metadata", {}) if isinstance(d, dict) else {})
        cid = md.get("doc_id") or md.get("chunk_id") or ""
        # source = "appointments" / "vitals" / ... — first token of cid before _
        src = cid.split("_")[0] if "_" in cid else ""
        yield cid, text, _parse_kv(text), src


def _get_index_or_reindex(pid: int):
    vs, _ = load_patient_index(pid)
    if vs is None or not getattr(vs, "documents", None):
        index_patient(pid)
        vs, _ = load_patient_index(pid)
    return vs


# ─── Per-rule selectors ──────────────────────────────────────────────────────

UPCOMING_STATUSES = {"confirmed", "pending", "scheduled"}
PAST_STATUSES = {"completed", "cancelled", "canceled", "no_show", "noshow"}


def pick_next_appointment(rows: List[Tuple[str, Dict[str, str]]]) -> List[str]:
    """Return [doc_id] of the soonest UPCOMING appointment (status ∈ confirmed/pending)."""
    cand = []
    for cid, kv in rows:
        st = kv.get("status", "").lower()
        sched = kv.get("scheduled_at", "")
        if st in UPCOMING_STATUSES and sched:
            cand.append((sched, cid))
    if not cand:
        return []
    cand.sort()  # ASC: earliest = "next"
    return [cand[0][1]]


def pick_last_appointment(rows: List[Tuple[str, Dict[str, str]]]) -> List[str]:
    """Return [doc_id] of the most-recent PAST appointment."""
    cand = []
    for cid, kv in rows:
        st = kv.get("status", "").lower()
        sched = kv.get("scheduled_at", "")
        if st in PAST_STATUSES and sched:
            cand.append((sched, cid))
    if not cand:
        # fall back: max date among all
        cand = [(kv.get("scheduled_at", ""), cid) for cid, kv in rows if kv.get("scheduled_at")]
    if not cand:
        return []
    cand.sort()
    return [cand[-1][1]]


def pick_latest_by_field(rows: List[Tuple[str, Dict[str, str]]], date_field: str) -> List[str]:
    cand = [(kv.get(date_field, ""), cid) for cid, kv in rows if kv.get(date_field)]
    if not cand:
        return []
    cand.sort()
    return [cand[-1][1]]


def pick_active_or_latest_med(rows: List[Tuple[str, Dict[str, str]]]) -> List[str]:
    """Active medications first, else most-recent prescribed."""
    active = [cid for cid, kv in rows if kv.get("status", "").lower() == "active"]
    if active:
        return active  # may legitimately be > 1 active medication
    return pick_latest_by_field(rows, "prescribed_at") or pick_latest_by_field(rows, "start_date")


# ─── Query → rule resolution ─────────────────────────────────────────────────

def resolve_rule(query: str, category: Optional[str]) -> Optional[Tuple[str, str]]:
    """
    Return (rule_name, source_filter) or None if query has no single-answer intent.

    rule_name ∈ {next_appointment, last_appointment, latest_vitals, latest_lab,
                  current_medication}
    source_filter is the chunk-id prefix to filter by ("appointments", "vitals", ...).
    """
    has_next = bool(NEXT_RX.search(query))
    has_last = bool(LAST_RX.search(query))
    has_current = bool(CURRENT_RX.search(query))

    if not (has_next or has_last or has_current):
        return None

    cat = (category or "").lower()

    if has_next and "appointment" in cat:
        return ("next_appointment", "appointments")
    if has_last and "appointment" in cat:
        return ("last_appointment", "appointments")
    if (has_last or has_current) and "vital" in cat:
        return ("latest_vitals", "vitals")
    if (has_last or has_current) and ("lab" in cat or "test" in cat):
        return ("latest_lab", "lab")  # chunk prefix is "lab" for lab_tests
    if has_current and "medication" in cat:
        return ("current_medication", "medications")
    if has_last and "medication" in cat:
        return ("current_medication", "medications")
    # English keywords + appointment/vitals/lab heuristic for queries with null category
    if "appointment" in query.lower() or "موعد" in query:
        if has_next:
            return ("next_appointment", "appointments")
        if has_last:
            return ("last_appointment", "appointments")
    return None


RULE_FN = {
    "next_appointment": pick_next_appointment,
    "last_appointment": pick_last_appointment,
    "latest_vitals": lambda r: pick_latest_by_field(r, "recorded_at"),
    "latest_lab": lambda r: pick_latest_by_field(r, "ordered_at") or pick_latest_by_field(r, "result_date"),
    "current_medication": pick_active_or_latest_med,
}


# ─── Driver ──────────────────────────────────────────────────────────────────

def main():
    v1 = json.loads(V1_PATH.read_text(encoding="utf-8"))
    print(f"loaded v1: {len(v1)} entries")

    # cache patient indexes
    pid_rows: Dict[int, Dict[str, List[Tuple[str, Dict[str, str]]]]] = {}

    def rows_for(pid: int, src_prefix: str) -> List[Tuple[str, Dict[str, str]]]:
        if pid not in pid_rows:
            vs = _get_index_or_reindex(pid)
            buckets: Dict[str, List[Tuple[str, Dict[str, str]]]] = defaultdict(list)
            if vs is not None:
                for cid, _txt, kv, src in _doc_iter(vs):
                    if src:
                        buckets[src].append((cid, kv))
            pid_rows[pid] = buckets
        return pid_rows[pid].get(src_prefix, [])

    v2 = []
    modified = 0
    skipped_no_rule = 0
    skipped_no_match = 0
    rule_counts: Dict[str, int] = defaultdict(int)

    for entry in v1:
        new_entry = dict(entry)
        rule = resolve_rule(entry["query"], entry.get("category"))
        if rule is None:
            skipped_no_rule += 1
            v2.append(new_entry)
            continue

        rule_name, src = rule
        rows = rows_for(entry["patient_id"], src)
        if not rows:
            skipped_no_match += 1
            v2.append(new_entry)
            continue

        picked = RULE_FN[rule_name](rows)
        if not picked:
            skipped_no_match += 1
            v2.append(new_entry)
            continue

        # Preserve any medical_records_* IDs from v1 as supporting context
        keep = [r for r in entry.get("relevant_ids", []) if r.startswith("medical_records_")]
        new_entry["relevant_ids"] = picked + [k for k in keep if k not in picked]
        new_entry["_v2_modified"] = True
        new_entry["_v2_rule"] = rule_name
        new_entry["_v2_v1_count"] = len(entry.get("relevant_ids", []))
        new_entry["_v2_v2_count"] = len(new_entry["relevant_ids"])
        modified += 1
        rule_counts[rule_name] += 1
        v2.append(new_entry)

    V2_PATH.write_text(json.dumps(v2, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote v2: {V2_PATH}")
    print(f"  total queries:        {len(v2)}")
    print(f"  modified:             {modified}")
    print(f"  unchanged (no rule):  {skipped_no_rule}")
    print(f"  unchanged (no match): {skipped_no_match}")
    print(f"  rule breakdown: {dict(rule_counts)}")

    # show a few samples
    print("\n=== sample modifications ===")
    shown = 0
    for e in v2:
        if e.get("_v2_modified"):
            print(f"  [{e['_v2_rule']}] '{e['query']}'")
            print(f"     v1 ids: {e['_v2_v1_count']}  ->  v2 ids: {e['_v2_v2_count']}")
            print(f"     v2 relevant: {e['relevant_ids']}")
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
