"""TOON Router — classifies patient queries into tiers and detects language.

FIXES vs original
─────────────────
L1  _ARABIC_RANGE loop replaced with re.search — O(1) vs O(256) per query.
L2  Removed ^ anchors from all T1 patterns — anchored regex rejects any query
    that doesn't start the string exactly, e.g. "can you tell me my blood type"
    silently falls through to T2 even though it's a T1 lookup.
L3  Added full Arabic keyword sets for all three tiers.
L4  keywords field now populated with the matched token for debugging.
L5  Removed unused uuid import.
L6  route() renamed to classify() so patient_rag_service.py imports work
    directly from this module. toon.py now delegates here instead of
    duplicating the logic.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List

# ─── Language detection ──────────────────────────────────────────────────────
# Single compiled pattern — checked once per query, not 256 chr() comparisons.
_AR_RE = re.compile(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]')


def detect_language(query: str) -> str:
    ar_chars = len(_AR_RE.findall(query))
    total = max(len(query.replace(" ", "")), 1)
    ratio = ar_chars / total
    if ratio > 0.5:
        return "ar"
    if ratio > 0.1:
        return "mixed"
    return "en"


# ─── Decision dataclass ──────────────────────────────────────────────────────

@dataclass
class RouterDecision:
    tier: int
    reason: str
    language: str
    keywords: List[str] = field(default_factory=list)


# ─── Pattern tables ──────────────────────────────────────────────────────────
# FIX L2: No ^ anchors. A patient asks "can you tell me my blood type?" —
# the ^ anchor kills that match. Medical queries are rarely the first word.
# FIX L3: Arabic patterns added for every tier.

# Tier 1 — single-field exact lookup. BM25 alone can answer these.
_T1_EN = [
    re.compile(r"what is my blood type",          re.I),
    re.compile(r"what.?s my (height|weight)",     re.I),
    re.compile(r"am i allergic|do i have allerg", re.I),
    re.compile(r"when is my (next|last) appointment", re.I),
    re.compile(r"how many prescription",          re.I),
    re.compile(r"show me my (vitals|appointments|results)", re.I),
    re.compile(r"how much do i owe",              re.I),
    re.compile(r"my (blood type|emergency contact)", re.I),
    re.compile(r"my insurance (number|provider)", re.I),
    re.compile(r"my (phone|email|address|dob|date of birth)", re.I),
]
# FIX: real Tier-1 queries are single-value lab/vital lookups, not just
# demographics. Match the "give me one current value" intent: level/ratio/
# value words, "نتائج فحص X", "هل فحص X طبيعي", single readings, bare lab codes.
_T1_AR = [
    # demographics
    re.compile(r"فصيلة دمي|ما فصيلة|نوع الدم", re.I),
    re.compile(r"طولي|وزني|كم وزن|كم طول",     re.I),
    re.compile(r"هل لدي حساسية|هل أنا حساس",    re.I),
    re.compile(r"موعدي القادم|آخر موعد",        re.I),
    re.compile(r"كم وصفة",                      re.I),
    re.compile(r"أريد مشاهدة (العلامات|المواعيد|النتائج)", re.I),
    re.compile(r"تأميني|رقم التأمين",            re.I),
    re.compile(r"رقم هاتفي|بريدي|عنواني",       re.I),
    # single lab / vital value lookups
    re.compile(r"مستوى|نسبة|قيمة|كثافة|حجم البول|عدد كريات|إنزيمات", re.I),
    re.compile(r"نتائج فحص|نتيجة (تحليل|فحص)|تحليل البول|فحص كامل|آخر فحص", re.I),
    re.compile(r"هل فحص .{1,14}طبيعي",          re.I),
    re.compile(r"قراءة|قراءاتي|سكر الدم|السكر في دمي", re.I),
    re.compile(r"فقر دم|بيلة",                  re.I),
    re.compile(r"النتائج الأولية|الرقم الهيدروجيني", re.I),
    # bare lab-code value lookups only when asked as a value (كم/ما),
    # NOT "هل X طبيعي؟" (gold treats those as T2).
    re.compile(r"(كم|ما)\s*(TSH|HbA1c|LDL|HDL|CBC|AST|ALT|FERITIN|free\s?T4)", re.I),
    # FIX (router precedence): T3 used to steal these single-field queries via
    # bare تشخيص / ملخص / توصيات / خطة العلاج. They are stored fields,
    # not reasoning queries, so anchor them in T1.
    re.compile(r"تشخيص حالتي|ملخص التشخيص|تشخيص مزمن", re.I),
    re.compile(r"خطة العلاج الموصوفة|توصيات الطبيب|ملاحظات الطبيب", re.I),
    re.compile(r"سبب زيارتي", re.I),  # reason_for_visit field
    # appointments single-field lookups
    re.compile(r"حالة موعدي|ملاحظات الموعد|موعد المتابعة", re.I),
    re.compile(r"الأعراض المسجلة في موعدي|موعدي الأخير كان", re.I),
    re.compile(r"تاريخ آخر زيارة|نوع السجل", re.I),
    # diagnosis / records single-field lookups
    re.compile(r"المرض المسجل|الحالة المرضية الأساسية|الشكوى الرئيسية", re.I),
    # medication single-field lookups
    re.compile(r"الأدوية الموصوفة|اسم الدواء|جرعة دوائي", re.I),
    re.compile(r"تعليمات (تناول )?الدواء|مدة (تناول )?الدواء", re.I),
    re.compile(r"كم مرة آخذ الدواء|متى صُرفت وصفتي|وصفتي الأخيرة", re.I),
    # vitals single-value lookups
    re.compile(r"درجة حرارتي|معدل ضربات القلب|معدل التنفس", re.I),
    # labs single-field lookups
    re.compile(r"الفحوصات المخبرية المطلوبة|اسم الفحص المطلوب|رمز الفحص", re.I),
    re.compile(r"تعليمات الفحص المخبري|متى تم طلب الفحص|حالة فحص فيتامين", re.I),
    re.compile(r"الفحص مكتمل أم ملغى", re.I),
    # billing single-value lookups
    re.compile(r"رسوم زيارتي|رسوم الكشف|كم دفعت|دفعت في آخر زيارة", re.I),
]

# Tier 2 — structured list/display. Hybrid retrieval, no LLM synthesis.
_T2_EN = [
    re.compile(r"recent|latest|last \d",    re.I),
    re.compile(r"(show|get|list|display) my (symptoms|diagnosis|prescription|lab|record|appointment)", re.I),
    re.compile(r"(what|which) (medication|medicine)",  re.I),
    re.compile(r"my medical (history|report|records?)", re.I),
    re.compile(r"(follow.?up|come back)",   re.I),
    re.compile(r"change in my (blood pressure|weight|glucose)", re.I),
    re.compile(r"my (appointments|prescriptions|lab (tests?|results?)|diagnos)", re.I),
    re.compile(r"medical.?records?",        re.I),
    re.compile(r"(upcoming|past|previous) (visit|appointment)", re.I),
]
# FIX: T2 is comparison / trend / range-judgment over time — NOT bare "آخر"
# (which wrongly stole single-value T1 lookups). Trigger on change verbs,
# comparison words, and "within normal range" phrasing.
_T2_AR = [
    re.compile(r"(اعرض|أظهر|قائمة) (الأعراض|التشخيص|الوصفات|المختبر|المواعيد)", re.I),
    re.compile(r"سجلاتي الطبية|تاريخي الطبي", re.I),
    re.compile(r"الزيارة القادمة", re.I),
    re.compile(r"وصفاتي|مواعيدي|نتائج مختبري", re.I),
    # comparison / trend / range-judgment
    re.compile(r"مقارنة|قارن",                re.I),
    re.compile(r"تحسّن|تحسن|تحسنت|حسّن|حسنت|تتحسن", re.I),
    re.compile(r"انخفض|ارتفع|ارتفعت|زاد|زادت|تراجع", re.I),
    re.compile(r"تغيّر|تغير|تغيرت|اتجاه",     re.I),
    re.compile(r"مستقر|مستقرة",               re.I),
    re.compile(r"ضمن (المعدل|النطاق)",         re.I),
    re.compile(r"مرتفع|منخفض",                re.I),
    re.compile(r"طبيعية",                      re.I),
    re.compile(r"منذ آخر",                     re.I),
    re.compile(r"قلق من النتائج|هناك قلق",     re.I),
    # FIX (router precedence): bare متابعة stole "متى موعد المتابعة؟"
    # (a single-field date lookup, T1). Bare فحوصات stole single-row
    # lab-test field lookups. Both removed; the explicit list/show-list
    # patterns above still cover genuine T2 multi-row queries.
]

# Tier 3 — requires LLM reasoning, synthesis, or clinical judgment.
_T3_EN = [
    re.compile(r"why (do i|does|is|are|did|should)", re.I),
    re.compile(r"what (cause|causes|caused|reason)",  re.I),
    re.compile(r"how (can i|could i|should i) (manage|deal with|prevent|treat|improve)", re.I),
    re.compile(r"(is it safe|should i worry|am i at risk|danger)",   re.I),
    re.compile(r"(should i|do i need to) (stop|avoid|continue|take)", re.I),
    re.compile(r"what (treatment|therapy|plan|next step)",            re.I),
    re.compile(r"(explain|meaning of|interpret) my (result|reading|level|report)", re.I),
    re.compile(r"is my (result|test|reading|level) (normal|high|low|elevated|concerning)", re.I),
    re.compile(r"my full (record|history|summary)",  re.I),
    re.compile(r"(drug|medication) interaction",      re.I),
    re.compile(r"summarize|summary|overview|analyse|analyze", re.I),
    re.compile(r"(risk|prognosis|outlook|long.?term)", re.I),
    re.compile(r"compare (my|the) (result|level|reading)", re.I),
    re.compile(r"based on my (labs|lab|results)", re.I),
    re.compile(r"lifestyle|do you recommend|what.*recommend|diet plan", re.I),
]
# FIX: T3 = clinical reasoning, recommendations, lifestyle/diet/exercise advice.
# Removed bare "تحليل" (it stole urinalysis lookups). Added the dominant
# reasoning triggers: "بناءً على", "هل يجب", "ما أفضل", "ما العلاقة",
# nutrition / exercise / sleep / cardiac-screening vocabulary.
_T3_AR = [
    re.compile(r"لماذا",                       re.I),
    re.compile(r"ما الذي يسبب",              re.I),
    re.compile(r"ما سبب (?:ارتفاع|انخفاض|الشعور|الألم|الصداع|الدوار|الإرهاق|التعب|الضعف|الغثيان|الحمى|السعال)", re.I),
    re.compile(r"بناءً على|بناء على|استناداً",  re.I),
    re.compile(r"كيف (يمكنني|أستطيع)|كيف أ",   re.I),
    re.compile(r"(هل هو آمن|هل يجب أن أقلق|هل أنا في خطر)", re.I),
    re.compile(r"هل يجب|يجب أن",               re.I),
    re.compile(r"ما أفضل|ما العلاقة",          re.I),
    re.compile(r"الخطة العلاجية|التغييرات المطلوبة|المسار المتوقع|التطورات", re.I),
    re.compile(r"(اشرح|فسّر|معنى) نتيجتي",      re.I),
    re.compile(r"هل نتيجتي (طبيعية|مرتفعة|منخفضة)", re.I),
    re.compile(r"تفاعل الأدوية|تداخل الأدوية",  re.I),
    # FIX (router precedence): require explicit summarise/synthesise verb;
    # bare ملخص / لخّص stole field-lookup queries like "ما ملخص التشخيص؟".
    re.compile(r"لخّص لي|أعطني ملخصاً|ملخص شامل", re.I),
    # FIX (router precedence): dropped bare تشخيص / خطة العلاج / توصيات
    # — those are field-lookups handled by T1. Kept genuine reasoning vocab.
    re.compile(r"خطر|الكشف المبكر|الفحوصات المطلوبة لتشخيص|قلق من ارتفاع", re.I),
    re.compile(r"نظام غذائي|التغذية|الأطعمة|الأطعة|مكملات|البروتين|الملح|الدهون|السمنة", re.I),
    re.compile(r"التمارين|التمرين|الرياضة", re.I),
    # FIX: bare نوم stole T2 "هل هناك دواء يؤخذ قبل النوم؟".
    # Match sleep only when paired with reasoning/advice intent.
    re.compile(r"كيف أنام|تحسين النوم|اضطراب النوم|أرق|جودة النوم", re.I),
    re.compile(r"ECG|الإيكوكارديوغرام|إيكو",    re.I),
    # FIX: extra reasoning vocabulary the original regex missed entirely.
    re.compile(r"الغرض العلاجي|الغرض من", re.I),
    re.compile(r"دلالة", re.I),
    re.compile(r"المضاعفات|مضاعفات", re.I),
    re.compile(r"النمط الزمني|نمط زمني", re.I),
    re.compile(r"ما أهمية|أهمية الفحوص", re.I),
    re.compile(r"تتوافق|تطابق|التوافق", re.I),
    re.compile(r"الإجراءات الوقائية|الوقاية", re.I),
    re.compile(r"ترابط|علاقة بين", re.I),
    re.compile(r"تأثير حالتي|تأثير.*على|نمط حياتي", re.I),
]


# ─── Classifier ─────────────────────────────────────────────────────────────
# Routing mode (env-overridable):
#   "hybrid" — learned bge-m3 classifier when confident, regex otherwise (default)
#   "ml"     — learned classifier only (falls back to regex if model unavailable)
#   "regex"  — legacy hand-tuned patterns only
_ROUTER_MODE = os.environ.get("TOON_ROUTER_MODE", "hybrid").lower()
_ML_CONF_THRESHOLD = float(os.environ.get("TOON_ROUTER_ML_THRESHOLD", "0.55"))
_ml_router = None
_ml_unavailable = False


def _get_ml_router():
    """Lazy, failure-tolerant load of the learned classifier.
    Never raises — if the model file or GPU is missing, returns None and the
    caller transparently falls back to regex (production never hard-breaks)."""
    global _ml_router, _ml_unavailable
    if _ml_router is None and not _ml_unavailable:
        try:
            from app.retrieval.toon_classifier import EmbeddingRouter

            _ml_router = EmbeddingRouter()
        except Exception:
            _ml_unavailable = True
            _ml_router = None
    return _ml_router


def classify(query: str, patient_id: int = 0) -> RouterDecision:
    """Route a query to a TOON tier.

    Strategy (production): use the learned bge-m3 classifier when it is
    confident; otherwise fall back to the deterministic regex router, which
    enforces the T3>T2>T1 clinical-safety ordering. Set TOON_ROUTER_MODE to
    override.
    """
    q = query.strip()
    if not q:
        return RouterDecision(tier=2, reason="empty_query", language="en")

    lang = detect_language(q)

    if _ROUTER_MODE in ("hybrid", "ml"):
        r = _get_ml_router()
        if r is not None:
            try:
                tier, conf = r.predict(q)
            except Exception:
                tier, conf = None, 0.0
            if tier is not None and (_ROUTER_MODE == "ml" or conf >= _ML_CONF_THRESHOLD):
                return RouterDecision(
                    tier=tier,
                    reason=f"ml_classifier(conf={conf:.2f})",
                    language=lang,
                    keywords=[],
                )
            # low confidence → defer to regex for a deterministic decision
        elif _ROUTER_MODE == "ml":
            return _classify_regex(q, lang)

    return _classify_regex(q, lang)


def _classify_regex(q: str, lang: str) -> RouterDecision:
    """
    Deterministic pattern router.
    T3 beats T2 beats T1 — never downgrade a reasoning query.
    """
    en_patterns = lang in ("en", "mixed")
    ar_patterns = lang in ("ar", "mixed")

    # T3 checked first — clinical safety demands we never short-circuit a
    # reasoning or synthesis query into a cheap tier.
    for patterns in ([_T3_EN] if en_patterns else []) + ([_T3_AR] if ar_patterns else []):
        for p in patterns:
            m = p.search(q)
            if m:
                return RouterDecision(
                    tier=3, reason="complex_reasoning",
                    language=lang, keywords=[m.group(0)]
                )

    for patterns in ([_T2_EN] if en_patterns else []) + ([_T2_AR] if ar_patterns else []):
        for p in patterns:
            m = p.search(q)
            if m:
                return RouterDecision(
                    tier=2, reason="semantic_request",
                    language=lang, keywords=[m.group(0)]
                )

    for patterns in ([_T1_EN] if en_patterns else []) + ([_T1_AR] if ar_patterns else []):
        for p in patterns:
            m = p.search(q)
            if m:
                return RouterDecision(
                    tier=1, reason="exact_field_lookup",
                    language=lang, keywords=[m.group(0)]
                )

    # Unknown intent — default to T2 (safe middle ground)
    return RouterDecision(tier=2, reason="default_tier2", language=lang)


# Alias so toon.py can import classify() from here without duplication
route = classify