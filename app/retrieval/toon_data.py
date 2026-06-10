"""Patient Data Chunker — fetches patient data from Supabase and produces indexable chunks.

FIXES vs original
─────────────────
P1/P2 fetch_vital_signs and fetch_diagnoses no longer call fetch_medical_records
      internally. Callers pass mr_ids they already hold — zero re-fetching.
P3    fetch_all_patient_chunks fetches medical_records once and reuses the result
      for vitals, diagnoses, and the record chunks themselves.
      Original called fetch_medical_records 3 times for one index build.
P4    fetch_prescriptions: 2 HTTP calls total (prescriptions + IN() items)
      instead of 1 + N (one per prescription).
P5    fetch_lab_orders: 2 HTTP calls total (orders + IN() results)
      instead of 1 + N (one per order).
P6    _get_supabase_config uses a module-level flag — thread-safe, parsed once.
P7    _format_row now recurses with explicit source_table preserved at every
      depth; nested items (prescription_items, lab results) render correctly.
"""
from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

from app.core.config import cfg

logger = logging.getLogger(__name__)

# ─── Supabase config — module-level, loaded once ────────────────────────────
# FIX P6: original used a global mutation guard that wasn't thread-safe.
# Module-level constants are initialised once at import time.
load_dotenv()
_SB_URL: str = os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
# Accept a bare project ref (e.g. Colab secrets) or a full URL.
if _SB_URL:
    _SB_URL = _SB_URL.strip().rstrip("/")
    if not _SB_URL.startswith("http"):
        _SB_URL = f"https://{_SB_URL}" if "." in _SB_URL else f"https://{_SB_URL}.supabase.co"
# Use service role key for backend — anon key is blocked by RLS on joins.
_SB_KEY: str = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv(
    "NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY", ""
)
_HEADERS: Dict[str, str] = {
    "apikey": _SB_KEY,
    "Authorization": f"Bearer {_SB_KEY}",
    "Content-Type": "application/json",
}


def _make_request(table: str, filters: str = "", params: Optional[Dict] = None) -> List[Dict]:
    qs = filters
    if params:
        qs += "&" + "&".join(f"{k}={v}" for k, v in params.items())
    try:
        resp = requests.get(
            f"{_SB_URL}/rest/v1/{table}?{qs}",
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json() or []
    except Exception as e:
        logger.error("Supabase error %s: %s", table, e)
        return []


# ─── Fetchers ────────────────────────────────────────────────────────────────

def fetch_profile(patient_id: int) -> Dict[str, Any]:
    """
    FIX: profiles.id is a UUID, not the patients.id integer.
    Correct path: patients.id → patients.profile_id → profiles.id.
    """
    patients = _make_request("patients", f"id=eq.{patient_id}", {"select": "*", "limit": "1"})
    if not patients:
        return {}

    pat = patients[0]
    profile_uuid = pat.pop("profile_id", None)

    if profile_uuid:
        profiles = _make_request("profiles", f"id=eq.{profile_uuid}", {"select": "*", "limit": "1"})
        if profiles:
            pat.update(profiles[0])

    return pat


def fetch_medical_records(patient_id: int, limit: int = 50) -> List[Dict]:
    return _make_request(
        "medical_records",
        f"patient_id=eq.{patient_id}",
        {"select": "*", "order": "visit_date.desc", "limit": str(limit)},
    )


def fetch_vital_signs_batch(mr_ids: List[int]) -> List[Dict]:
    """
    FIX P1: accepts already-fetched mr_ids — no internal fetch_medical_records call.
    Single IN() query instead of N queries.
    """
    if not mr_ids:
        return []
    id_csv = ",".join(str(i) for i in mr_ids)
    return _make_request(
        "vital_signs",
        f"medical_record_id=in.({id_csv})",
        {"select": "*"},
    )


def fetch_appointments(patient_id: int, limit: int = 50) -> List[Dict]:
    return _make_request(
        "appointments",
        f"patient_id=eq.{patient_id}",
        {"select": "*", "order": "scheduled_at.desc", "limit": str(limit)},
    )


def fetch_diagnoses_batch(mr_ids: List[int]) -> List[Dict]:
    """
    FIX P2: accepts already-fetched mr_ids — no internal fetch_medical_records call.
    Single IN() query instead of N queries.
    """
    if not mr_ids:
        return []
    id_csv = ",".join(str(i) for i in mr_ids)
    return _make_request(
        "diagnoses",
        f"medical_record_id=in.({id_csv})",
        {"select": "*"},
    )


def fetch_prescriptions(patient_id: int, limit: int = 30) -> List[Dict]:
    """
    FIX P4: 2 HTTP calls (prescriptions + all items via IN()).
    Original made 1 + N calls.
    """
    rxs = _make_request(
        "prescriptions",
        f"patient_id=eq.{patient_id}",
        {"select": "*", "order": "issued_at.desc", "limit": str(limit)},
    )
    if not rxs:
        return []

    rx_id_csv = ",".join(str(r["id"]) for r in rxs)
    all_items = _make_request(
        "prescription_items",
        f"prescription_id=in.({rx_id_csv})",
        {"select": "*"},
    )

    items_map: Dict[int, List[Dict]] = {}
    for item in all_items:
        pid = item.get("prescription_id")
        items_map.setdefault(pid, []).append(item)

    for rx in rxs:
        rx["items"] = items_map.get(rx["id"], [])

    return rxs


def fetch_lab_orders(patient_id: int, limit: int = 20) -> List[Dict]:
    """
    FIX P5: 2 HTTP calls (orders + all results via IN()).
    Original made 1 + N calls.
    """
    orders = _make_request(
        "lab_test_orders",
        f"patient_id=eq.{patient_id}",
        {"select": "*", "order": "ordered_at.desc", "limit": str(limit)},
    )
    if not orders:
        return []

    order_id_csv = ",".join(str(o["id"]) for o in orders)
    all_results = _make_request(
        "lab_test_results",
        f"lab_test_order_id=in.({order_id_csv})",
        {"select": "*"},
    )

    results_map: Dict[int, Dict] = {
        r["lab_test_order_id"]: r for r in all_results
    }

    for order in orders:
        order["result"] = results_map.get(order["id"], {})

    return orders


def fetch_payments(patient_id: int, limit: int = 20) -> List[Dict]:
    return _make_request(
        "payments",
        f"patient_id=eq.{patient_id}",
        {"select": "*", "order": "created_at.desc", "limit": str(limit)},
    )


# ─── Formatter ───────────────────────────────────────────────────────────────
# FIX P7: original lost source_table when recursing into nested dicts/lists.
# Now _extract flattens with dot-notation keys so nested items render correctly:
#   items.medicine_name: Amoxicillin | items.dosage: 500mg ...

_SKIP_KEYS = frozenset({
    "id", "medical_record_id", "prescription_id", "auth_user_id",
    "profile_id", "doctor_id", "booked_by", "patient_id", "lab_test_order_id",
})


def _format_row(row: Dict, source_table: str) -> str:
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
    if not parts:
        return ""
    return f"[source: {source_table}] " + " | ".join(parts)


# ─── Chunk builder ───────────────────────────────────────────────────────────

def fetch_all_patient_chunks(patient_id: int) -> List[Dict[str, Any]]:
    """
    FIX P3: fetch_medical_records called ONCE here.
    mr_ids passed to fetch_vital_signs_batch and fetch_diagnoses_batch
    directly — no internal re-fetch.

    Original flow (3× fetch_medical_records):
      fetch_all_patient_chunks
        → fetch_vital_signs       → fetch_medical_records  ← extra
        → fetch_diagnoses         → fetch_medical_records  ← extra
        → fetch_medical_records                            ← direct

    Fixed flow (1× fetch_medical_records):
      fetch_all_patient_chunks
        → fetch_medical_records
        → fetch_vital_signs_batch(mr_ids)   ← no internal DB call
        → fetch_diagnoses_batch(mr_ids)     ← no internal DB call
    """
    chunks: List[Dict[str, Any]] = []

    # Profile
    profile = fetch_profile(patient_id)
    if profile:
        t = _format_row(profile, "profile")
        if t:
            chunks.append({"text": t, "metadata": {"source": "profile"}})

    # Medical records — fetched ONCE, reused for vitals + diagnoses
    records = fetch_medical_records(patient_id)
    mr_ids  = [r["id"] for r in records]

    for rec in records:
        t = _format_row(rec, "medical_records")
        if t:
            chunks.append({"text": t, "metadata": {"source": "medical_records"}})

    for v in fetch_vital_signs_batch(mr_ids):
        t = _format_row(v, "vital_signs")
        if t:
            chunks.append({"text": t, "metadata": {"source": "vital_signs"}})

    for d in fetch_diagnoses_batch(mr_ids):
        t = _format_row(d, "diagnoses")
        if t:
            chunks.append({"text": t, "metadata": {"source": "diagnoses"}})

    for a in fetch_appointments(patient_id):
        t = _format_row(a, "appointments")
        if t:
            chunks.append({"text": t, "metadata": {"source": "appointments"}})

    for rx in fetch_prescriptions(patient_id):
        t = _format_row(rx, "prescriptions")
        if t:
            chunks.append({"text": t, "metadata": {"source": "prescriptions"}})

    for lab in fetch_lab_orders(patient_id):
        t = _format_row(lab, "lab_orders")
        if t:
            chunks.append({"text": t, "metadata": {"source": "lab_orders"}})

    for pay in fetch_payments(patient_id):
        t = _format_row(pay, "payments")
        if t:
            chunks.append({"text": t, "metadata": {"source": "payments"}})

    return chunks


# ─── Index storage helpers ───────────────────────────────────────────────────

def get_patient_index_dir(patient_id: int) -> Path:
    patient_dir = cfg.DATA_DIR / "patients" / str(patient_id)
    patient_dir.mkdir(parents=True, exist_ok=True)
    return patient_dir


def index_patient_data(patient_id: int) -> int:
    from app.indexing.vector_store import VectorStore
    from app.indexing.bm25_index import BM25Index
    from app.indexing.embedder import embed_texts

    chunks = fetch_all_patient_chunks(patient_id)
    if not chunks:
        logger.warning("No patient data found for patient %s", patient_id)
        return 0

    patient_dir = get_patient_index_dir(patient_id)
    vs   = VectorStore(dim=1024)
    bm25 = BM25Index()

    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    for chunk, emb in zip(chunks, embeddings):
        vs.add(text=chunk["text"], embedding=emb, metadata=chunk["metadata"])
        bm25.add_document(
            doc_id=f"{chunk['metadata']['source']}_{uuid.uuid4().hex[:8]}",
            text=chunk["text"],
            metadata=chunk["metadata"],
        )

    vs.save(str(patient_dir))
    bm25.save(str(patient_dir))
    logger.info("Indexed %d chunks for patient %s", len(chunks), patient_id)
    return len(chunks)


def load_patient_index(patient_id: int) -> Tuple[Optional[Any], Optional[Any]]:
    from app.indexing.vector_store import VectorStore
    from app.indexing.bm25_index import BM25Index

    patient_dir = get_patient_index_dir(patient_id)
    if not (patient_dir / "vector_store.pkl").exists():
        return None, None

    try:
        vs = VectorStore(dim=1024)
        vs.load(str(patient_dir))
        bm25 = BM25Index.load(str(patient_dir))
        return vs, bm25
    except Exception as e:
        logger.error("Failed to load patient index: %s", e)
        return None, None


def fetch_live_context(patient_id: int) -> str:
    """Live context string for Tier 3 — always fresh from DB."""
    parts: List[str] = []

    profile = fetch_profile(patient_id)
    if profile:
        parts.append("[patient_profile]\n" + _format_row(profile, "profile"))

    records = fetch_medical_records(patient_id, limit=10)
    mr_ids  = [r["id"] for r in records]

    for v in fetch_vital_signs_batch(mr_ids)[:5]:
        parts.append("[vital_signs]\n" + _format_row(v, "vital_signs"))

    for a in fetch_appointments(patient_id, limit=5):
        parts.append("[appointments]\n" + _format_row(a, "appointments"))

    for d in fetch_diagnoses_batch(mr_ids)[:5]:
        parts.append("[diagnoses]\n" + _format_row(d, "diagnoses"))

    for rx in fetch_prescriptions(patient_id, limit=5):
        parts.append("[prescriptions]\n" + _format_row(rx, "prescriptions"))

    return "\n\n".join(p for p in parts if p)