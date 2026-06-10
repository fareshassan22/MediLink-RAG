"""TOON — Token-Optimized Orchestration Network for MediLink.
Patient Data retrieval layer with 3-tier routing.
"""
from __future__ import annotations

import logging
import os
import re
import uuid
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from app.core.config import cfg

logger = logging.getLogger(__name__)


def _get_supabase() -> Tuple[str, str]:
    load_dotenv()
    url = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "https://icntpbdznkfajnieyrjq.supabase.co")
    key = os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", "")
    return _normalize_sb_url(url), key


def _normalize_sb_url(url: str) -> str:
    """Accept a full URL or a bare project ref and return a usable base URL.

    Some environments (e.g. Colab secrets) store only the project ref
    ('icntpbdznkfajnieyrjq') instead of the full 'https://<ref>.supabase.co'.
    """
    url = (url or "").strip().rstrip("/")
    if not url:
        return url
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if "." in url:                       # has a domain but no scheme
        return f"https://{url}"
    return f"https://{url}.supabase.co"  # bare project ref


_SB_URL, _SB_KEY = _get_supabase()
_SB_HEADERS = {
    "apikey": _SB_KEY,
    "Authorization": f"Bearer {_SB_KEY}",
    "Content-Type": "application/json",
}


def _sb_get(table: str, qs: str) -> List[Dict]:
    """Single Supabase REST call. qs is the full query string after '?'."""
    try:
        resp = requests.get(
            f"{_SB_URL}/rest/v1/{table}?{qs}",
            headers=_SB_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json() or []
    except Exception as e:
        logger.error("Supabase %s: %s", table, e)
        return []
 
 
# ─── Tier routing — delegates to toon_router ────────────────────────────────
 
TOKEN_BUDGETS = {1: 50, 2: 200, 3: 20_000}
 
 
# ─── Token-budget enforcement ────────────────────────────────────────────────
# PRODUCTION FIX: retrieval used to return raw concatenated rows that blew past
# the per-tier budget (T1 326/50, T2 686/200). The cheap tiers were not cheap.
# These helpers pack WHOLE rows up to the budget (keeping each row intact) and
# hard-truncate only if a single row already exceeds the budget.

_TIKTOKEN_ENC = None
_TIKTOKEN_TRIED = False


def _get_encoder():
    global _TIKTOKEN_ENC, _TIKTOKEN_TRIED
    if not _TIKTOKEN_TRIED:
        _TIKTOKEN_TRIED = True
        try:
            import tiktoken

            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENC = None
    return _TIKTOKEN_ENC


def _count_tokens(text: str) -> int:
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text.split()))  # fallback: word count


def _truncate_tokens(text: str, budget: int) -> str:
    enc = _get_encoder()
    if enc is not None:
        toks = enc.encode(text)
        return text if len(toks) <= budget else enc.decode(toks[:budget])
    words = text.split()
    return text if len(words) <= budget else " ".join(words[:budget])


def _pack_to_budget(texts: List[str], budget: Optional[int]) -> str:
    """Concatenate whole rows up to `budget` tokens (rows kept intact).
    If the first row alone exceeds the budget, hard-truncate it."""
    texts = [t for t in texts if t]
    if budget is None:
        return "\n\n".join(texts)
    sep_tokens = _count_tokens("\n\n")
    out: List[str] = []
    used = 0
    for t in texts:
        n = _count_tokens(t)
        if not out and n > budget:
            return _truncate_tokens(t, budget)
        added = n + (sep_tokens if out else 0)
        if used + added > budget:
            break
        out.append(t)
        used += added
    return "\n\n".join(out)


def _dense_rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    """PRODUCTION FIX (Tier-1 ranking): BM25 alone ranked the correct Arabic
    row too low (Recall@1 = 0). Re-rank BM25 candidates by bge-m3 dense cosine
    similarity (strong multilingual/Arabic) so the right row surfaces at rank 1.
    Patient row sets are tiny (~26), so this is cheap."""
    if not candidates:
        return candidates
    try:
        import numpy as np

        from app.indexing.embedder import embed_texts

        texts = [c.get("text", "") for c in candidates]
        embs = embed_texts([query] + texts)
        q = embs[0]
        sims = [float(np.dot(q, e)) for e in embs[1:]]
        order = sorted(range(len(candidates)), key=lambda i: sims[i], reverse=True)
        return [candidates[i] for i in order]
    except Exception as e:
        logger.warning("TOON dense rerank failed, using BM25 order: %s", e)
        return candidates


# ─── Cross-encoder reranker (highest-ROI ranking model) ──────────────────────
# A bi-encoder (bge-m3 dense cosine in _dense_rerank) scores query and row
# independently. A cross-encoder reads (query, row) jointly and is markedly
# better at putting the truly relevant row at rank 1. bge-reranker-v2-m3 is
# multilingual (strong Arabic). Loaded lazily once and cached. Disable with
# TOON_CROSS_ENCODER=0 to fall back to dense rerank.
_CROSS_ENCODER = None
_CROSS_ENCODER_TRIED = False


def _get_cross_encoder():
    global _CROSS_ENCODER, _CROSS_ENCODER_TRIED
    if not _CROSS_ENCODER_TRIED:
        _CROSS_ENCODER_TRIED = True
        if os.environ.get("TOON_CROSS_ENCODER", "1") == "0":
            return None
        try:
            import torch
            from sentence_transformers import CrossEncoder

            model_name = os.environ.get("TOON_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _CROSS_ENCODER = CrossEncoder(model_name, device=device, max_length=512)
            logger.info("TOON cross-encoder loaded: %s on %s", model_name, device)
        except Exception as e:
            logger.warning("Cross-encoder unavailable, using dense rerank: %s", e)
            _CROSS_ENCODER = None
    return _CROSS_ENCODER


def _pool_size(default: int, corpus_n: int) -> int:
    """Candidate-pool depth for reranking. TOON_POOL_SIZE controls it:
      0  (default) -> use the whole patient corpus (rank every chunk),
      N>0          -> cap the pool at N candidates,
    falling back to `default` when the corpus size is unknown."""
    try:
        env = int(os.environ.get("TOON_POOL_SIZE", "0") or 0)
    except ValueError:
        env = 0
    if env > 0:
        return env if corpus_n <= 0 else min(env, corpus_n)
    if corpus_n > 0:
        return corpus_n
    return default


def _cross_encoder_rerank(query: str, candidates: List[Dict]) -> List[Dict]:
    """Rerank candidate row dicts (each with a 'text' key) by joint
    cross-encoder relevance. Falls back to dense rerank, then to the input
    order, on any failure."""
    if not candidates:
        return candidates
    ce = _get_cross_encoder()
    if ce is None:
        return _dense_rerank(query, candidates)
    try:
        pairs = [(query, c.get("text", "")) for c in candidates]
        scores = ce.predict(pairs)
        order = sorted(range(len(candidates)), key=lambda i: float(scores[i]), reverse=True)
        return [candidates[i] for i in order]
    except Exception as e:
        logger.warning("Cross-encoder rerank failed, using dense rerank: %s", e)
        return _dense_rerank(query, candidates)


# ─── Date-aware reranking for appointment "next/last" intent ─────────────────
# The cross-encoder ranks rows by semantic similarity, but every appointment
# row looks near-identical to it (same fields, same words). What makes one the
# "next" or "last" appointment is the DATE, which the cross-encoder cannot
# reason about. This step parses scheduled_at from dated candidates and moves
# the chronologically-correct row to rank 1. The rule (next = earliest upcoming,
# last = most recent past, relative to today) is objectively correct and does
# NOT depend on any ground-truth labels.

_NEXT_INTENT = re.compile(
    r"(next|upcoming|coming|future|soonest|القادم|القادمة|القادمه|التالي|"
    r"التالية|المقبل|المقبلة|القادمين)", re.IGNORECASE)
_LAST_INTENT = re.compile(
    r"(last|previous|recent|latest|most recent|past|prior|السابق|السابقة|"
    r"الأخير|الأخيرة|الاخير|الاخيرة|آخر|اخر|الماضي|الماضية|الفائت)", re.IGNORECASE)
_FIRST_INTENT = re.compile(
    r"(first|earliest|initial|very first|أول|اول|الأول|الأولى|الاول|الاولى)",
    re.IGNORECASE)
_APPT_HINT = re.compile(
    r"(appointment|appt|visit|booking|موعد|مواعيد|زيار|الحجز|حجز)",
    re.IGNORECASE)
_SCHED_RE = re.compile(
    r"scheduled_at:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}(?:[T ][0-9:]{4,8})?)")
_STATUS_RE = re.compile(r"status:\s*([a-z_]+)", re.IGNORECASE)


def _parse_sched(text: str) -> Optional[datetime]:
    m = _SCHED_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1).replace("T", " ").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _appt_status(text: str) -> Optional[str]:
    m = _STATUS_RE.search(text or "")
    return m.group(1).lower() if m else None


def _apply_date_intent(query: str, candidates: List[Dict]) -> List[Dict]:
    """Promote the chronologically-correct appointment row to rank 1 when the
    query asks for the next/last appointment. No-op for any other query.

    Definitions (objective, label-independent):
      next  = the earliest appointment still in the future.
      last  = the most recent *completed* visit (a visit that actually happened);
              falls back to the most recent past slot if none are marked completed.
      first = the earliest appointment on record (the patient's first visit).
    Using status=completed for "last" matches the natural meaning of "last visit"
    (آخر زيارة) — a future-but-unattended slot is not a past visit."""
    if not candidates:
        return candidates
    want_next = bool(_NEXT_INTENT.search(query))
    want_last = bool(_LAST_INTENT.search(query))
    want_first = bool(_FIRST_INTENT.search(query))
    # Need exactly one clear temporal intent, and an appointment context.
    n_intents = want_next + want_last + want_first
    if n_intents != 1:                   # neither, or ambiguous combination
        return candidates
    if not _APPT_HINT.search(query):
        return candidates

    dated = [
        (i, _parse_sched(c.get("text", "")), _appt_status(c.get("text", "")))
        for i, c in enumerate(candidates)
    ]
    dated = [(i, dt, st) for i, dt, st in dated if dt is not None]
    if not dated:
        return candidates

    now = datetime.now()
    if want_next:
        future = [(i, dt) for i, dt, _ in dated if dt >= now]
        target = (min(future, key=lambda x: x[1])[0] if future
                  else max(dated, key=lambda x: x[1])[0])
    elif want_first:                     # earliest appointment on record
        target = min(dated, key=lambda x: x[1])[0]
    else:  # want_last — most recent visit that actually happened
        completed = [(i, dt) for i, dt, st in dated if st == "completed" and dt <= now]
        past = [(i, dt) for i, dt, _ in dated if dt <= now]
        if completed:
            target = max(completed, key=lambda x: x[1])[0]
        elif past:
            target = max(past, key=lambda x: x[1])[0]
        else:
            target = min(dated, key=lambda x: x[1])[0]

    return [candidates[target]] + [c for j, c in enumerate(candidates) if j != target]


@dataclass
class TierDecision:
    tier: int
    reason: str
    language: str
    keywords: List[str] = field(default_factory=list)


def classify(query: str, patient_id: int = 0) -> TierDecision:
    # Delegate to the canonical router — no duplicate logic.
    from app.retrieval.toon_router import classify as _route
    d = _route(query, patient_id)
    return TierDecision(
        tier=d.tier, reason=d.reason,
        language=d.language, keywords=d.keywords,
    )
 
 
# ─── Column whitelists ───────────────────────────────────────────────────────
 
_SKIP_KEYS = frozenset({
    "id", "medical_record_id", "prescription_id", "auth_user_id",
    "profile_id", "doctor_id", "booked_by", "patient_id",
})
 
_PROFILE_COLS  = "full_name,email,phone,gender,date_of_birth,city,address_line,country"
_PATIENT_COLS  = ("profile_id,blood_type,height_cm,weight_kg,allergies,chronic_diseases,"
                  "emergency_contact_name,emergency_contact_phone,"
                  "insurance_provider,insurance_number")
_VITAL_COLS    = ("temperature_c,blood_pressure_systolic,blood_pressure_diastolic,"
                  "heart_rate,oxygen_saturation,bmi,blood_glucose,"
                  "respiratory_rate,weight_kg,height_cm")
_APPT_COLS     = "scheduled_at,status,reason_for_visit,symptoms,visit_fee,notes,is_first_visit"
_DIAG_COLS     = "diagnosis_name,icd_code,is_primary,notes,medical_record_id"
_RX_COLS       = "id,notes,issued_at,patient_id"
_ITEM_COLS     = "medicine_name,dosage,frequency,duration,instructions,route,quantity,prescription_id"
_LAB_COLS      = "id,test_name,test_code,status,ordered_at,instructions,patient_id"
_RESULT_COLS   = "result_summary,completed_at,lab_test_order_id"
_MR_COLS       = ("id,visit_date,record_type,chief_complaint,diagnosis_summary,"
                  "treatment_plan,doctor_notes,follow_up_date,patient_id")
 
 
# ─── Fetchers ────────────────────────────────────────────────────────────────

def patient_exists(patient_id: int) -> bool:
    """Return True iff the patient has any accessible records.

    The ``patients`` table is RLS-protected and returns empty under the
    publishable key, so existence is determined from the data tables the API
    can actually read (medical_records, then appointments).

    Raises ConnectionError if the database cannot be reached, so callers can
    return 503 instead of a misleading 404 during an outage.
    """
    for table in ("medical_records", "appointments"):
        try:
            resp = requests.get(
                f"{_SB_URL}/rest/v1/{table}?patient_id=eq.{patient_id}&select=id&limit=1",
                headers=_SB_HEADERS,
                timeout=10,
            )
            resp.raise_for_status()
        except Exception as e:
            raise ConnectionError(f"patient lookup failed: {e}") from e
        if resp.json():
            return True
    return False


def fetch_profile(patient_id: int) -> Dict:
    """
    FIX C2: profiles.id is a UUID (= auth.users.id), NOT the patients.id integer.
    Correct join path: patients.id → patients.profile_id → profiles.id (UUID).
    """
    patients = _sb_get("patients", f"id=eq.{patient_id}&select={_PATIENT_COLS}&limit=1")
    if not patients:
        return {}
    pat = patients[0]
    profile_uuid = pat.pop("profile_id", None)
    if profile_uuid:
        profiles = _sb_get("profiles", f"id=eq.{profile_uuid}&select={_PROFILE_COLS}&limit=1")
        if profiles:
            pat.update(profiles[0])
    return pat
 
 
def fetch_medical_records(patient_id: int, limit: int = 50) -> List[Dict]:
    return _sb_get(
        "medical_records",
        f"patient_id=eq.{patient_id}&select={_MR_COLS}&order=visit_date.desc&limit={limit}",
    )
 
 
def fetch_vitals_batch(mr_ids: List[int]) -> List[Dict]:
    """
    FIX C4: single IN() call instead of one HTTP request per medical record.
    Pass the list of medical_record ids from an already-fetched records list.
    """
    if not mr_ids:
        return []
    id_csv = ",".join(str(i) for i in mr_ids)
    return _sb_get(
        "vital_signs",
        f"medical_record_id=in.({id_csv})&select={_VITAL_COLS},medical_record_id",
    )
 
 
def fetch_diagnoses_batch(mr_ids: List[int]) -> List[Dict]:
    """
    FIX C3: single IN() call instead of one HTTP request per medical record.
    """
    if not mr_ids:
        return []
    id_csv = ",".join(str(i) for i in mr_ids)
    return _sb_get(
        "diagnoses",
        f"medical_record_id=in.({id_csv})&select={_DIAG_COLS}",
    )
 
 
def fetch_appointments(patient_id: int, limit: int = 30) -> List[Dict]:
    return _sb_get(
        "appointments",
        f"patient_id=eq.{patient_id}&select={_APPT_COLS}&order=scheduled_at.desc&limit={limit}",
    )
 
 
def fetch_prescriptions_batch(patient_id: int, limit: int = 20) -> List[Dict]:
    """
    FIX: fetch all prescriptions then all items in exactly 2 HTTP calls,
    not 1 + N calls (one per prescription).
    """
    rxs = _sb_get(
        "prescriptions",
        f"patient_id=eq.{patient_id}&select={_RX_COLS}&order=issued_at.desc&limit={limit}",
    )
    if not rxs:
        return []
 
    rx_ids = ",".join(str(r["id"]) for r in rxs)
    items  = _sb_get(
        "prescription_items",
        f"prescription_id=in.({rx_ids})&select={_ITEM_COLS}",
    )
    # Group items by prescription_id
    items_map: Dict[int, List[Dict]] = {}
    for item in items:
        pid = item.pop("prescription_id", None)
        items_map.setdefault(pid, []).append(item)
 
    for rx in rxs:
        rx["items"] = items_map.get(rx["id"], [])
    return rxs
 
 
def fetch_lab_orders_batch(patient_id: int, limit: int = 20) -> List[Dict]:
    """
    FIX: fetch all orders then all results in 2 HTTP calls, not 1 + N.
    """
    orders = _sb_get(
        "lab_test_orders",
        f"patient_id=eq.{patient_id}&select={_LAB_COLS}&order=ordered_at.desc&limit={limit}",
    )
    if not orders:
        return []
 
    order_ids = ",".join(str(o["id"]) for o in orders)
    results   = _sb_get(
        "lab_test_results",
        f"lab_test_order_id=in.({order_ids})&select={_RESULT_COLS}",
    )
    results_map = {r.pop("lab_test_order_id"): r for r in results}
 
    for order in orders:
        order["result"] = results_map.get(order["id"], {})
    return orders
 
 
def fetch_payments(patient_id: int, limit: int = 20) -> List[Dict]:
    return _sb_get(
        "payments",
        f"patient_id=eq.{patient_id}&select=amount,currency,payment_method,status,paid_at"
        f"&order=created_at.desc&limit={limit}",
    )
 
 
# ─── Formatter ───────────────────────────────────────────────────────────────
 
def _fmt(row: Dict, src: str) -> str:
    parts: List[str] = []
 
    def _extract(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in _SKIP_KEYS or v is None or v == "" or v == []:
                    continue
                label = f"{prefix}{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    _extract(v, prefix=f"{label}.")
                else:
                    parts.append(f"{label}: {v}")
        elif isinstance(obj, list):
            for item in obj:
                _extract(item, prefix)
 
    _extract(row)
    return f"[source: {src}] " + " | ".join(parts) if parts else ""
 
 
# ─── Live context for Tier 3 ─────────────────────────────────────────────────
 
def fetch_live_context(patient_id: int) -> str:
    """
    Fetches all tables in the minimum number of HTTP calls (batch pattern).
    Records are fetched once and shared between diagnoses/vitals lookups.
    """
    sections: List[str] = []
 
    profile = fetch_profile(patient_id)
    if profile:
        sections.append("[patient_profile]\n" + _fmt(profile, "profile"))
 
    records = fetch_medical_records(patient_id, limit=10)
    mr_ids  = [r["id"] for r in records]
 
    for rec in records:
        sections.append("[medical_record]\n" + _fmt(rec, "medical_records"))
 
    for v in fetch_vitals_batch(mr_ids)[:5]:
        sections.append("[vital_signs]\n" + _fmt(v, "vital_signs"))
 
    for a in fetch_appointments(patient_id, limit=5):
        sections.append("[appointment]\n" + _fmt(a, "appointments"))
 
    for d in fetch_diagnoses_batch(mr_ids)[:5]:
        sections.append("[diagnosis]\n" + _fmt(d, "diagnoses"))
 
    for rx in fetch_prescriptions_batch(patient_id, limit=5):
        sections.append("[prescription]\n" + _fmt(rx, "prescriptions"))
 
    for lab in fetch_lab_orders_batch(patient_id, limit=5):
        sections.append("[lab_test]\n" + _fmt(lab, "lab_test_orders"))
 
    return "\n\n".join(s for s in sections if s)
 
 
# ─── Full clinical history for doctor access ─────────────────────────────────
 
def fetch_full_context(patient_id: int, max_tokens: Optional[int] = None) -> str:
    """
    Doctor-facing full history. Unlike fetch_live_context (which caps each table
    to 5–10 rows for the patient view), this returns the COMPLETE record set —
    all medical records, vitals, diagnoses, appointments, prescriptions, lab
    results, and payments — assembled as one text block for LLM consumption.

    When `max_tokens` is set, sections are emitted in clinical-priority order
    (profile → diagnoses → records → prescriptions → labs → vitals →
    appointments → payments) and packed to the budget, so the most important
    history survives truncation. This keeps the request under hosted-LLM
    tokens-per-minute limits (e.g. Groq free tier = 6000 TPM).
    """
    profile_sec: List[str] = []
    diag_sec: List[str] = []
    record_sec: List[str] = []
    rx_sec: List[str] = []
    lab_sec: List[str] = []
    vital_sec: List[str] = []
    appt_sec: List[str] = []
    pay_sec: List[str] = []

    profile = fetch_profile(patient_id)
    if profile:
        profile_sec.append("[patient_profile]\n" + _fmt(profile, "profile"))

    records = fetch_medical_records(patient_id)
    mr_ids  = [r["id"] for r in records]

    for rec in records:
        record_sec.append("[medical_record]\n" + _fmt(rec, "medical_records"))

    for v in fetch_vitals_batch(mr_ids):
        vital_sec.append("[vital_signs]\n" + _fmt(v, "vital_signs"))

    for d in fetch_diagnoses_batch(mr_ids):
        diag_sec.append("[diagnosis]\n" + _fmt(d, "diagnoses"))

    for a in fetch_appointments(patient_id):
        appt_sec.append("[appointment]\n" + _fmt(a, "appointments"))

    for rx in fetch_prescriptions_batch(patient_id):
        rx_sec.append("[prescription]\n" + _fmt(rx, "prescriptions"))

    for lab in fetch_lab_orders_batch(patient_id):
        lab_sec.append("[lab_test]\n" + _fmt(lab, "lab_test_orders"))

    for pay in fetch_payments(patient_id):
        pay_sec.append("[payment]\n" + _fmt(pay, "payments"))

    # Clinical-priority order for budget packing.
    sections = (
        profile_sec + diag_sec + record_sec + rx_sec
        + lab_sec + vital_sec + appt_sec + pay_sec
    )
    sections = [s for s in sections if s]

    if max_tokens is not None:
        return _pack_to_budget(sections, max_tokens)
    return "\n\n".join(sections)
 
 
# ─── Chunk builder for indexing ──────────────────────────────────────────────
 
def _doc_id(table: str, row: Dict) -> str:
    """Deterministic, stable doc_id for a Supabase row.

    Random UUIDs (the old scheme) change on every re-index, which makes any
    ground-truth file built against them rot. A stable id keeps retrieval
    metrics (recall/MRR/NDCG) and the index cache valid across re-indexing.

    Tables with a primary key (medical_records, prescriptions) use it directly.
    Tables without one (vital_signs, diagnoses, appointments) fall back to a
    short content hash of the row — stable as long as the row data is stable.
    """
    rid = row.get("id") or row.get("uuid") or row.get(f"{table}_id")
    if rid is not None:
        return f"{table}_{rid}"
    canonical = json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.md5(canonical.encode("utf-8")).hexdigest()[:10]
    return f"{table}_{digest}"


def fetch_all_chunks(patient_id: int) -> List[Dict]:
    """
    FIX P3: fetch_medical_records called once here and passed down —
    original code called it 3× (once direct + once inside fetch_vitals +
    once inside fetch_diagnoses).
    """
    chunks: List[Dict] = []
 
    profile = fetch_profile(patient_id)
    if profile:
        t = _fmt(profile, "profile")
        if t:
            chunks.append({"text": t, "metadata": {"table": "profile", "doc_id": f"profile_{patient_id}"}})
 
    records = fetch_medical_records(patient_id)
    mr_ids  = [r["id"] for r in records]
 
    for rec in records:
        t = _fmt(rec, "medical_records")
        if t:
            chunks.append({"text": t, "metadata": {"table": "medical_records", "doc_id": _doc_id("medical_records", rec)}})
 
    for v in fetch_vitals_batch(mr_ids):
        t = _fmt(v, "vital_signs")
        if t:
            chunks.append({"text": t, "metadata": {"table": "vital_signs", "doc_id": _doc_id("vital_signs", v)}})
 
    for d in fetch_diagnoses_batch(mr_ids):
        t = _fmt(d, "diagnoses")
        if t:
            chunks.append({"text": t, "metadata": {"table": "diagnoses", "doc_id": _doc_id("diagnoses", d)}})
 
    for a in fetch_appointments(patient_id):
        t = _fmt(a, "appointments")
        if t:
            chunks.append({"text": t, "metadata": {"table": "appointments", "doc_id": _doc_id("appointments", a)}})
 
    for rx in fetch_prescriptions_batch(patient_id):
        t = _fmt(rx, "prescriptions")
        if t:
            chunks.append({"text": t, "metadata": {"table": "prescriptions", "doc_id": _doc_id("prescriptions", rx)}})
 
    for lab in fetch_lab_orders_batch(patient_id):
        t = _fmt(lab, "lab_test_orders")
        if t:
            chunks.append({"text": t, "metadata": {"table": "lab_test_orders", "doc_id": _doc_id("lab_test_orders", lab)}})
 
    for pay in fetch_payments(patient_id):
        t = _fmt(pay, "payments")
        if t:
            chunks.append({"text": t, "metadata": {"table": "payments", "doc_id": _doc_id("payments", pay)}})
 
    return chunks
 
 
# ─── In-memory index cache — the 45 s latency fix ────────────────────────────
# FIX C5/C6: original search_bm25 and search_hybrid both called
# load_patient_index() which reads from disk AND re-initialised the embedder
# (bge-m3, 570 MB) on every single request. Now the index lives in this dict
# for the lifetime of the Uvicorn worker process.
_INDEX_CACHE: Dict[int, Dict[str, Any]] = {}
 
 
def _patient_dir(patient_id: int) -> Path:
    d = cfg.DATA_DIR / "patients" / str(patient_id)
    d.mkdir(parents=True, exist_ok=True)
    return d
 
 
def load_patient_index(patient_id: int) -> Tuple[Optional[Any], Optional[Any]]:
    # Serve from memory first
    if patient_id in _INDEX_CACHE:
        e = _INDEX_CACHE[patient_id]
        return e["vs"], e["bm25"]
 
    # Cold load from disk into cache
    d = _patient_dir(patient_id)
    if not (d / "vector_store.pkl").exists():
        return None, None
 
    try:
        from app.indexing.vector_store import VectorStore
        from app.indexing.bm25_index import BM25Index
 
        vs = VectorStore(dim=1024)
        vs.load(str(d))
        bm25 = BM25Index.load(str(d))
 
        _INDEX_CACHE[patient_id] = {"vs": vs, "bm25": bm25}
        logger.info("TOON: loaded index for patient %s from disk → cached", patient_id)
        return vs, bm25
    except Exception as e:
        logger.error("Failed to load patient index: %s", e)
        return None, None
 
 
def index_patient(patient_id: int) -> int:
    from app.indexing.vector_store import VectorStore
    from app.indexing.bm25_index import BM25Index
    from app.indexing.embedder import embed_texts
 
    chunks = fetch_all_chunks(patient_id)
    if not chunks:
        return 0
 
    vs   = VectorStore(dim=1024)
    bm25 = BM25Index()
    texts = [c["text"] for c in chunks]
    embs  = embed_texts(texts)
 
    for chunk, emb in zip(chunks, embs):
        vs.add(text=chunk["text"], embedding=emb, metadata=chunk["metadata"])
        bm25.add_document(
            doc_id=chunk["metadata"]["doc_id"],
            text=chunk["text"],
            metadata=chunk["metadata"],
        )
 
    d = _patient_dir(patient_id)
    vs.save(str(d))
    bm25.save(str(d))
 
    # Write to cache so the next request is instant
    _INDEX_CACHE[patient_id] = {"vs": vs, "bm25": bm25}
    logger.info("TOON: indexed + cached %d chunks for patient %s", len(chunks), patient_id)
    return len(chunks)
 
 
# ─── Tier 1 — BM25 search ────────────────────────────────────────────────────

def search_bm25(patient_id: int, query: str, top_k: int = 5, return_ids: bool = False,
                token_budget: Optional[int] = None, rerank: bool = True):
    # FIX C5: reads from _INDEX_CACHE — no disk reload.
    vs, bm25 = load_patient_index(patient_id)
    if not bm25:
        return [] if return_ids else ""
    # Over-fetch a DEEP candidate pool, then cross-encoder rerank so the
    # truly relevant row lands at rank 1 (a row BM25 buried at rank 25 can
    # only be recovered if it is inside the pool we rerank). Most patient
    # records hold 50-100 chunks, so a fixed pool of 30 hides ~half the
    # corpus from the reranker; TOON_POOL_SIZE (default 0 = use the whole
    # patient corpus) lets the cross-encoder see every candidate.
    pool_size = _pool_size(max(top_k * 5, 30), len(getattr(bm25, "documents", []) or []))
    results = bm25.search(query, k=pool_size)
    if rerank and results:
        results = _cross_encoder_rerank(query, results)
    results = _apply_date_intent(query, results)
    results = results[:top_k]
    if return_ids:
        return [r["metadata"].get("doc_id", "") for r in results if r.get("metadata")]
    texts = [r["text"] for r in results if r.get("text")]
    return _pack_to_budget(texts, token_budget)


# ─── Tier 2 — Hybrid search with real RRF ────────────────────────────────────

def search_hybrid(patient_id: int, query: str, top_k: int = 10, return_ids: bool = False,
                  token_budget: Optional[int] = None):
    """
    FIX C6: reads index from cache — no bge-m3 reload.
    FIX C7: proper Reciprocal Rank Fusion replaces broken dedup-concat.

    RRF score = Σ 1/(k + rank_i) for each retriever i.
    k=60 is the standard constant (Cormack et al. 2009).
    """
    vs, bm25 = load_patient_index(patient_id)
    if not vs and not bm25:
        return [] if return_ids else ""

    from app.indexing.embedder import embed_texts

    # Map text -> doc_id so RRF (which fuses on text) can emit stable ids.
    text_to_id: Dict[str, str] = {}

    # Deeper per-retriever pools give the fusion + cross-encoder more true
    # positives to promote (you cannot rerank a row you never fetched).
    # Default (TOON_POOL_SIZE=0) widens the pool to the whole patient corpus
    # so the cross-encoder ranks every chunk, not just the first 30.
    corpus_n = max(len(getattr(vs, "documents", []) or []),
                   len(getattr(bm25, "documents", []) or []))
    pool_each = _pool_size(max(top_k * 3, 30), corpus_n)

    # Dense results
    dense_texts: List[str] = []
    if vs:
        embs         = embed_texts([query])
        dense_results = vs.search(embs[0], k=pool_each)
        dense_texts   = [d["text"] for d in dense_results if d.get("text")]
        for d in dense_results:
            if d.get("text") and d.get("metadata"):
                text_to_id[d["text"]] = d["metadata"].get("doc_id", "")

    # Sparse results
    bm25_texts: List[str] = []
    if bm25:
        bm25_results = bm25.search(query, k=pool_each)
        bm25_texts   = [d["text"] for d in bm25_results if d.get("text")]
        for d in bm25_results:
            if d.get("text") and d.get("metadata"):
                text_to_id.setdefault(d["text"], d["metadata"].get("doc_id", ""))

    # FIX C7: Reciprocal Rank Fusion
    rrf_k = 60
    scores: Dict[str, float] = {}

    for rank, text in enumerate(dense_texts):
        scores[text] = scores.get(text, 0.0) + 1.0 / (rrf_k + rank + 1)

    for rank, text in enumerate(bm25_texts):
        scores[text] = scores.get(text, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Cross-encoder rerank the fused candidate pool (joint query+row scoring),
    # then keep top_k. RRF gives a strong recall-oriented pool; the cross-encoder
    # sharpens precision at the top.
    pool = [{"text": text, "metadata": {"doc_id": text_to_id.get(text, "")}}
            for text, _ in ranked[:pool_each]]
    pool = _cross_encoder_rerank(query, pool)
    pool = _apply_date_intent(query, pool)
    top = pool[:top_k]

    if return_ids:
        return [r["metadata"].get("doc_id", "") for r in top if r.get("metadata")]
    return _pack_to_budget([r["text"] for r in top if r.get("text")], token_budget)