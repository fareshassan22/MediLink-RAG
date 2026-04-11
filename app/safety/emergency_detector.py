from app.utils.arabic import normalize_arabic as _normalize


# ---------- Emergency phrases (Arabic) ----------
EMERGENCY_KEYWORDS_AR = [
    "لا استطيع التنفس",
    "نزيف حاد",
    "الم شديد جدا في الصدر",
    "الم شديد في الصدر",
    "الم في الصدر",
    "فقدان الوعي",
    "تشنجات",
    "ضيق تنفس حاد",
    "ضيق في التنفس",
    "نوبه قلبيه",
    "جلطه قلبيه",
    "جلطه دماغيه",
    "سكته دماغيه",
    "تسمم",
    "حساسيه شديده",
    "صدمه تحسسيه",
    "اختناق",
    "حروق شديده",
    "كسر مفتوح",
    "نزيف داخلي",
    "فقدان الوعي المفاجئ",
    "توقف القلب",
]

# ---------- Emergency phrases (English) ----------
EMERGENCY_KEYWORDS_EN = [
    "can't breathe",
    "cannot breathe",
    "difficulty breathing",
    "severe bleeding",
    "loss of consciousness",
    "seizure",
    "heart attack",
    "stroke",
    "anaphylaxis",
    "choking",
    "poisoning",
    "severe chest pain",
    "chest pain",
    "cardiac arrest",
    "severe burn",
    "open fracture",
    "internal bleeding",
    "suicidal",
    "overdose",
]

# Pre-normalise Arabic keywords once at import time
_NORM_AR = [_normalize(kw) for kw in EMERGENCY_KEYWORDS_AR]
_LOWER_EN = [kw.lower() for kw in EMERGENCY_KEYWORDS_EN]


def detect_emergency(query: str) -> bool:
    """Detect emergency phrases with Arabic normalization and English support."""
    if not query:
        return False
    norm = _normalize(query)
    lower = query.lower()
    return any(kw in norm for kw in _NORM_AR) or any(kw in lower for kw in _LOWER_EN)
