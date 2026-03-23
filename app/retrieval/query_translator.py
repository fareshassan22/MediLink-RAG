"""Query translation module for bilingual (Arabic/English) retrieval.

Uses Groq LLM API for Arabic→English translation (fast, no local model),
with a dictionary fallback if the API is unavailable.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# ── Arabic normalizer (same rules as preprocessing.py) ───────

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


# ── Groq API translation ─────────────────────────────────────

_groq_client = None


def _get_groq_client():
    """Lazy-load the Groq client (reuses the same API key as generation)."""
    global _groq_client
    if _groq_client is not None:
        return _groq_client

    from dotenv import load_dotenv
    from groq import Groq

    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    _groq_client = Groq(api_key=api_key)
    return _groq_client


def _api_translate(query: str) -> str:
    """Translate Arabic→English using Groq API. Returns empty string on failure."""
    try:
        client = _get_groq_client()
        if client is None:
            return ""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Translate the following Arabic medical query to English. "
                        "Return ONLY the English translation, nothing else. "
                        "Do not add explanations or notes."
                    ),
                },
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=100,
        )
        result = response.choices[0].message.content.strip()
        # Strip quotes if the model wraps the translation
        if result.startswith('"') and result.endswith('"'):
            result = result[1:-1].strip()
        return result if result and any(c.isalpha() for c in result) else ""
    except Exception as e:
        logger.warning("Groq translation failed: %s", e)
        return ""


# ── Dictionary fallback ──────────────────────────────────────

_ARABIC_TO_ENGLISH = {
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


def _dictionary_translate(query: str) -> str:
    """Translate Arabic query using the hardcoded dictionary (fallback)."""
    translated = _norm(query)

    sorted_terms = sorted(
        _ARABIC_TO_ENGLISH.items(), key=lambda x: len(x[0]), reverse=True
    )

    for arabic, english in sorted_terms:
        norm_key = _norm(arabic)
        translated = translated.replace(norm_key, english)

    translated = re.sub(r"and([a-z])", r"and \1", translated)
    translated = re.sub(r"(\w)([A-Z])", r"\1 \2", translated)
    translated = " ".join(translated.split())

    return translated


# ── Public API ────────────────────────────────────────────────


def translate_query(query: str) -> str:
    """Translate Arabic query to English for cross-lingual retrieval.

    Uses Groq API (llama-3.1-8b-instant) as the primary translator.
    Falls back to dictionary-based translation if the API fails.
    """
    if not query or not query.strip():
        return query

    # Try API-based translation first
    api_result = _api_translate(query)
    if api_result:
        return api_result

    # Fallback to dictionary
    return _dictionary_translate(query)


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
