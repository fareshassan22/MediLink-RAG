#!/usr/bin/env python3
"""Build a BALANCED MULTI-PATIENT TOON query set (data-aware assignment).

The original benchmark hardcoded patient 200 and was ~half lab-assay queries, so
it never exercised appointments, prescriptions, diagnoses, records, vitals or
billing retrieval. This version uses a hand-written BALANCED query bank that
spans every Supabase table, tags each query with its data category + tier, and
assigns it to a patient that actually holds that data — grounding the ground
truth in real rows WITHOUT faking anything (retrieval still has to find them).

Outputs:
    data/toon_multipatient_queries.json   [{"query","tier","patient_id","category"}]
    results/lab_seeding_report.json        lab orders queried but with empty result
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

from app.retrieval import toon

OUTPUT_PATH = "data/toon_multipatient_queries.json"
LAB_REPORT_PATH = "results/lab_seeding_report.json"

T1, T2, T3 = "tier_1_simple", "tier_2_moderate", "tier_3_complex"

# --- query category -> the corpus table that can answer it -------------------
CATEGORY_TABLE = {
    "vitals": "vital_signs",
    "labs": "lab_test_orders",
    "meds": "prescriptions",
    "appointments": "appointments",
    "diagnosis": "diagnoses",
    "records": "medical_records",
    "billing": "payments",
}

# --- balanced query bank: (query, tier, category) ----------------------------
QUERY_BANK = [
    # appointments (14)
    ("متى موعدي القادم؟", T1, "appointments"),
    ("ما سبب زيارتي الأخيرة؟", T1, "appointments"),
    ("ما حالة موعدي الحالي؟", T1, "appointments"),
    ("ما الأعراض المسجلة في موعدي؟", T1, "appointments"),
    ("هل موعدي الأخير كان أول زيارة؟", T1, "appointments"),
    ("ما ملاحظات الموعد؟", T1, "appointments"),
    ("متى كان آخر موعد لي؟", T1, "appointments"),
    ("كم عدد مواعيدي المكتملة؟", T2, "appointments"),
    ("هل لدي مواعيد معلقة؟", T2, "appointments"),
    ("هل لدي مواعيد ملغاة؟", T2, "appointments"),
    ("ما أكثر سبب متكرر لزياراتي؟", T2, "appointments"),
    ("بناءً على مواعيدي السابقة، متى يجب أن أحجز المتابعة القادمة؟", T3, "appointments"),
    ("بناءً على أعراضي في المواعيد، هل حالتي تتطور؟", T3, "appointments"),
    ("ما النمط الزمني لزياراتي للعيادة؟", T3, "appointments"),
    # medications (14)
    ("ما الأدوية الموصوفة لي؟", T1, "meds"),
    ("ما جرعة دوائي؟", T1, "meds"),
    ("كم مرة آخذ الدواء في اليوم؟", T1, "meds"),
    ("ما مدة تناول الدواء؟", T1, "meds"),
    ("ما تعليمات تناول الدواء؟", T1, "meds"),
    ("ما اسم الدواء الموصوف؟", T1, "meds"),
    ("متى صُرفت وصفتي الأخيرة؟", T1, "meds"),
    ("هل تغيرت أدويتي مؤخراً؟", T2, "meds"),
    ("ما الأدوية التي أتناولها بانتظام؟", T2, "meds"),
    ("هل هناك دواء يؤخذ قبل النوم؟", T2, "meds"),
    ("كم عدد الأدوية في وصفتي؟", T2, "meds"),
    ("بناءً على أدويتي الحالية، هل هناك تعارض محتمل؟", T3, "meds"),
    ("هل يجب تعديل جرعة دوائي بناءً على حالتي؟", T3, "meds"),
    ("ما الغرض العلاجي من الأدوية الموصوفة؟", T3, "meds"),
    # diagnosis (11)
    ("ما تشخيص حالتي؟", T1, "diagnosis"),
    ("ما المرض المسجل في ملفي؟", T1, "diagnosis"),
    ("ما الشكوى الرئيسية لدي؟", T1, "diagnosis"),
    ("ما ملخص التشخيص؟", T1, "diagnosis"),
    ("هل لدي تشخيص مزمن؟", T1, "diagnosis"),
    ("ما الحالة المرضية الأساسية؟", T1, "diagnosis"),
    ("هل تغير تشخيصي مع الوقت؟", T2, "diagnosis"),
    ("ما التشخيصات المتكررة في سجلي؟", T2, "diagnosis"),
    ("هل هناك أكثر من تشخيص؟", T2, "diagnosis"),
    ("بناءً على تشخيصي، ما الخطة العلاجية المناسبة؟", T3, "diagnosis"),
    ("ما المضاعفات المحتملة لتشخيصي؟", T3, "diagnosis"),
    # medical records (11)
    ("ما ملاحظات الطبيب في آخر زيارة؟", T1, "records"),
    ("ما خطة العلاج الموصوفة؟", T1, "records"),
    ("متى موعد المتابعة؟", T1, "records"),
    ("ما نوع السجل الطبي؟", T1, "records"),
    ("ما تاريخ آخر زيارة طبية؟", T1, "records"),
    ("ما توصيات الطبيب؟", T1, "records"),
    ("ما ملخص زياراتي الطبية؟", T2, "records"),
    ("كيف تطورت حالتي عبر الزيارات؟", T2, "records"),
    ("ما خطط العلاج عبر سجلي؟", T2, "records"),
    ("بناءً على سجلي الطبي الكامل، ما التوصيات؟", T3, "records"),
    ("ما المسار العلاجي بناءً على سجلاتي؟", T3, "records"),
    # vitals (13)
    ("ما مستوى السكر في دمي؟", T1, "vitals"),
    ("ما قراءة ضغط الدم لدي؟", T1, "vitals"),
    ("كم وزني في آخر فحص؟", T1, "vitals"),
    ("ما درجة حرارتي؟", T1, "vitals"),
    ("ما معدل ضربات القلب؟", T1, "vitals"),
    ("ما معدل التنفس؟", T1, "vitals"),
    ("ما نسبة الأكسجين في الدم؟", T1, "vitals"),
    ("هل تحسنت قراءات الضغط؟", T2, "vitals"),
    ("هل هناك تغير في وزني؟", T2, "vitals"),
    ("هل مستوى السكر مستقر؟", T2, "vitals"),
    ("ما اتجاه مؤشراتي الحيوية؟", T2, "vitals"),
    ("بناءً على مؤشراتي الحيوية، هل حالتي مستقرة؟", T3, "vitals"),
    ("ما دلالة قراءاتي الحيوية على صحتي؟", T3, "vitals"),
    # labs (13)
    ("ما الفحوصات المخبرية المطلوبة لي؟", T1, "labs"),
    ("ما حالة فحص فيتامين د؟", T1, "labs"),
    ("ما تعليمات الفحص المخبري؟", T1, "labs"),
    ("متى تم طلب الفحص؟", T1, "labs"),
    ("ما اسم الفحص المطلوب؟", T1, "labs"),
    ("ما رمز الفحص؟", T1, "labs"),
    ("هل الفحص مكتمل أم ملغى؟", T1, "labs"),
    ("هل اكتملت جميع فحوصاتي؟", T2, "labs"),
    ("كم عدد الفحوصات المطلوبة؟", T2, "labs"),
    ("ما الفحوصات الملغاة؟", T2, "labs"),
    ("هل هناك فحوصات معلقة؟", T2, "labs"),
    ("بناءً على الفحوصات المطلوبة، ماذا يجب أن أتابع؟", T3, "labs"),
    ("ما أهمية الفحوصات المطلوبة لحالتي؟", T3, "labs"),
    # billing (8)
    ("كم رسوم زيارتي؟", T1, "billing"),
    ("ما تكلفة آخر موعد؟", T1, "billing"),
    ("ما رسوم الكشف؟", T1, "billing"),
    ("كم دفعت في آخر زيارة؟", T1, "billing"),
    ("ما قيمة الفاتورة؟", T1, "billing"),
    ("كم إجمالي رسوم زياراتي؟", T2, "billing"),
    ("ما متوسط تكلفة زياراتي؟", T2, "billing"),
    ("هل تختلف رسوم زياراتي؟", T2, "billing"),
    # multi / cross-record reasoning (16, all tier-3)
    ("بناءً على حالتي العامة، ما النصائح الصحية المناسبة؟", T3, "multi"),
    ("ما العلاقة بين تشخيصي وأدويتي؟", T3, "multi"),
    ("بناءً على سجلي، ما نمط الحياة الموصى به؟", T3, "multi"),
    ("هل أعراضي تتوافق مع تشخيصي؟", T3, "multi"),
    ("ما أفضل نظام غذائي بناءً على حالتي؟", T3, "multi"),
    ("بناءً على مواعيدي وأدويتي، هل ألتزم بالعلاج؟", T3, "multi"),
    ("ما التمارين المناسبة لحالتي؟", T3, "multi"),
    ("كيف أحسّن حالتي الصحية العامة؟", T3, "multi"),
    ("بناءً على بياناتي، ما عوامل الخطر لدي؟", T3, "multi"),
    ("هل يجب تغيير خطة علاجي؟", T3, "multi"),
    ("ما الإجراءات الوقائية المناسبة لحالتي؟", T3, "multi"),
    ("بناءً على تاريخي الطبي، ما المتابعة المطلوبة؟", T3, "multi"),
    ("ما تأثير حالتي على نمط حياتي اليومي؟", T3, "multi"),
    ("هل هناك ترابط بين أعراضي المختلفة؟", T3, "multi"),
    ("ما الأولويات العلاجية بناءً على حالتي؟", T3, "multi"),
    ("بناءً على كل بياناتي، ما ملخص حالتي الصحية؟", T3, "multi"),
]


def patient_table_presence():
    """For every patient, which corpus tables actually contain rows + row counts."""
    pids = set()
    for tbl in ("medical_records", "appointments", "prescriptions", "lab_test_orders"):
        for r in toon._sb_get(tbl, "select=patient_id"):
            if r.get("patient_id") is not None:
                pids.add(int(r["patient_id"]))

    presence: dict[int, set] = {}
    counts: dict[int, int] = {}
    for pid in sorted(pids):
        tables, n = set(), 0
        for ch in toon.fetch_all_chunks(pid):
            tables.add(ch["metadata"].get("table"))
            n += 1
        presence[pid] = tables
        counts[pid] = n
    return presence, counts


def main():
    presence, counts = patient_table_presence()
    pids = sorted(presence)
    print(f"{len(pids)} patients with data")
    print(f"{len(QUERY_BANK)} queries in balanced bank")
    cat_hist = Counter(c for _, _, c in QUERY_BANK)
    print("query categories:", dict(cat_hist))

    richest = sorted(pids, key=lambda p: counts[p], reverse=True)

    def pool_for(category: str) -> list:
        if category == "multi":
            return richest[: max(10, len(richest) // 2)]
        table = CATEGORY_TABLE[category]
        elig = [p for p in pids if table in presence[p]]
        return elig or richest

    # Round-robin within each category's eligible pool: balanced load AND every
    # query lands on a patient that can plausibly answer it.
    rr: dict[str, int] = defaultdict(int)
    assigned = []
    for query, tier, category in QUERY_BANK:
        pool = pool_for(category)
        pid = pool[rr[category] % len(pool)]
        rr[category] += 1
        assigned.append({
            "query": query, "tier": tier, "patient_id": pid, "category": category,
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(assigned, f, ensure_ascii=False, indent=2)
    used = sorted({a["patient_id"] for a in assigned})
    print(f"Wrote {len(assigned)} queries spanning {len(used)} patients -> {OUTPUT_PATH}")
    print("tier mix:", dict(Counter(a["tier"] for a in assigned)))

    # --- lab-seeding report (data gap we will NOT fake) ----------------------
    lab_rows = []
    for pid in pids:
        try:
            lab_rows.extend(toon.fetch_lab_orders_batch(pid))
        except Exception:
            pass
    by_code = Counter()
    empty = 0
    for lr in lab_rows:
        code = lr.get("test_code") or lr.get("test_name") or "UNKNOWN"
        by_code[code] += 1
        if not lr.get("result"):
            empty += 1
    report = {
        "summary": (
            f"{empty}/{len(lab_rows)} lab orders in Supabase have an EMPTY result "
            f"payload. Lab status/metadata queries are answerable, but exact assay "
            f"VALUES require seeding these results."
        ),
        "total_lab_orders": len(lab_rows),
        "lab_orders_with_empty_result": empty,
        "distinct_test_codes": dict(by_code.most_common()),
    }
    os.makedirs("results", exist_ok=True)
    with open(LAB_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Lab-seeding report -> {LAB_REPORT_PATH}")
    print("  " + report["summary"])


if __name__ == "__main__":
    main()
