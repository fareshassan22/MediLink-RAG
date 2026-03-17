"""Query translation module for bilingual (Arabic/English) retrieval."""

import re

# Lightweight Arabic normalizer (same rules as preprocessing.py)
_ALEF = re.compile("[إأآا]")
_YA = re.compile("[ىي]")
_TA = re.compile("[ة]")
_DIACRITICS = re.compile(r"[\u064B-\u0652]")
_TATWEEL = re.compile("ـ")


def _norm(text: str) -> str:
    text = _ALEF.sub("ا", text)
    text = _YA.sub("ي", text)
    text = _TA.sub("ه", text)
    text = _DIACRITICS.sub("", text)
    text = _TATWEEL.sub("", text)
    return text


ARABIC_TO_ENGLISH = {
    # Questions - full phrases first
    "ما العلاقة بين": "what is the relationship between",
    "ما هو": "what is",
    "ما هي": "what are",
    "ما هو علاج": "what is the treatment for",
    "ما هي أعراض": "what are the symptoms of",
    "ما هي أسباب": "what are the causes of",
    "ما علاج": "what is treatment for",
    "ما هو الفرق بين": "what is the difference between",
    "ما العلاقة": "what is the relationship",
    "كيف يمكن": "how can",
    "كيف يتم": "how is",
    "كيف": "how",
    "لماذا": "why",
    "هل": "is",
    # Medical terms
    "أعراض": "symptoms",
    "أسباب": "causes",
    "علاج": "treatment",
    "تشخيص": "diagnosis",
    "وقاية": "prevention",
    "مرض": "disease",
    "أمراض": "diseases",
    "القلب": "heart",
    "السكري": "diabetes",
    "السرطن": "cancer",
    "العدوى": "infection",
    "ارتفاع ضغط الدم": "hypertension",
    "ضغط الدم المرتفع": "high blood pressure",
    "ضغط الدم": "blood pressure",
    "ارتفاع ضغط": "hypertension",
    "أمراض القلب": "heart disease",
    "العلاقة": "relationship",
    "العلاقة بين": "relationship between",
    "بين": "between",
    "و": "and",
    "الربو": "asthma",
    "الأنيميا": "anemia",
    "فقر الدم": "anemia",
    "فقر": "anemia",
    "فييتأمين د": "vitamin D",
    "فييتامين د": "vitamin D",
    "فييامين د": "vitamin D",
    "فييتأمين": "vitamin",
    "فييتامين": "vitamin",
    "فييامين": "vitamin",
    "vitamin د": "vitamin D",
    "نقص": "deficiency",
    # Body parts
    "قلب": "heart",
    "كبد": "liver",
    "كلية": "kidney",
    "رئة": "lung",
    "معدة": "stomach",
    "دم": "blood",
    "عظم": "bone",
    "مفصل": "joint",
    "عضلة": "muscle",
    "جلد": "skin",
    "عين": "eye",
    "غدة": "gland",
    "غدة درقية": "thyroid",
    # Diseases
    "سكري": "diabetes",
    "ضغط": "blood pressure",
    "أمراض القلب": "heart disease",
    "علاقة": "relationship",
    "سرطان": "cancer",
    "ورم": "tumor",
    "عدوى": "infection",
    "التهاب": "inflammation",
    "ربو": "asthma",
    "فقر الدم": "anemia",
    "أنيميا": "anemia",
    "فييتأمين د": "vitamin D",
    "فييتامين د": "vitamin D",
    "فييامين د": "vitamin D",
    "فييتأمين": "vitamin",
    "فييتامين": "vitamin",
    "فييامين": "vitamin",
    "vitamin د": "vitamin D",
    "نقص": "deficiency",
    "أنفلونزا": "influenza",
    # Treatments
    "جراحة": "surgery",
    "عملية": "operation",
    "دواء": "medication",
    "أدوية": "medications",
    "أنسولين": "insulin",
    "مضاد حيوي": "antibiotic",
    # Emergency
    "طارئ": "emergency",
    "جلطة": "clot",
    "جلطة قلبية": "heart attack",
    "سكتة": "stroke",
    "نزيف": "bleeding",
}


def translate_query(query: str) -> str:
    """Translate Arabic query terms to English for cross-lingual retrieval."""
    # Normalise Arabic characters so variants match the dictionary
    translated = _norm(query)

    # Pre-normalise dictionary keys once (done each call — small dict, fast enough)
    sorted_terms = sorted(
        ARABIC_TO_ENGLISH.items(), key=lambda x: len(x[0]), reverse=True
    )

    for arabic, english in sorted_terms:
        norm_key = _norm(arabic)
        translated = translated.replace(norm_key, english)

    # Fix spacing issues (e.g., "andheart" -> "and heart")
    import re

    # Fix known pattern: "and" followed by word
    translated = re.sub(r"and([a-z])", r"and \1", translated)

    # Fix CamelCase-like patterns
    translated = re.sub(r"(\w)([A-Z])", r"\1 \2", translated)

    # Clean up extra spaces
    translated = " ".join(translated.split())

    return translated


def translate_query_bilingual(query: str) -> str:
    """Return both original and translated query combined."""
    translated = translate_query(query)

    if translated != query:
        return f"{query} {translated}"
    return query


def is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    if not text:
        return False
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    total_chars = len(re.findall(r"[\w]", text))
    if total_chars == 0:
        return False
    return arabic_chars / total_chars > 0.3
