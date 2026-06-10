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

# ---------- Co-occurrence rules (tolerate intervening words) ----------
# Plain substring matching misses phrases like "ألم حاد في الصدر" because the
# severity word ("حاد") sits between the keyword tokens. Each rule is a list of
# token-groups; the rule fires only when EVERY group has at least one token
# present in the text — so "chest" + "pain" anywhere together escalates,
# regardless of the words in between.
_COOCCUR_AR = [
    [("صدر",), ("الم", "وجع")],                                  # chest + pain
    [("ضيق", "صعوب"), ("تنفس",)],                                # tightness/difficulty + breathing
    [("مفاج",), ("تشويش", "رؤي", "بصر", "نطق", "كلام", "شلل", "خدر")],  # sudden + neuro sign (stroke)
]
_COOCCUR_EN = [
    [("chest",), ("pain", "tightness")],
    [("breath", "breathing"),
     ("short", "shortness", "difficulty", "trouble", "hard", "can't", "cannot")],
    [("sudden",), ("vision", "blurred", "slurred", "speech", "numbness", "weakness")],
]


def _matches_cooccurrence(text: str, rules) -> bool:
    return any(
        all(any(tok in text for tok in group) for group in rule)
        for rule in rules
    )


def detect_emergency(query: str) -> bool:
    """Detect emergency phrases with Arabic normalization and English support."""
    if not query:
        return False
    norm = _normalize(query)
    lower = query.lower()
    if any(kw in norm for kw in _NORM_AR) or any(kw in lower for kw in _LOWER_EN):
        return True
    return _matches_cooccurrence(norm, _COOCCUR_AR) or _matches_cooccurrence(lower, _COOCCUR_EN)
