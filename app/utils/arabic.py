"""Shared Arabic text utilities used across the project.

Single source of truth for Arabic normalization and language detection.
"""

import re

ARABIC_ALEF_VARIATIONS = re.compile("[إأآا]")
ARABIC_YA_VARIATIONS = re.compile("[ىي]")
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652]")
ARABIC_TATWEEL = re.compile("ـ")


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text by standardizing character variations.

    Handles:
    - Alef variations (إأآا → ا)
    - Ya variations (ىي → ي)
    - Diacritics removal
    - Tatweel (ـ) removal

    Note: Ta marbuta (ة) is intentionally NOT normalized.
    Mapping ة→ه destroys medical terms (e.g. صحة→صحه).
    """
    if not isinstance(text, str):
        text = str(text)
    text = ARABIC_ALEF_VARIATIONS.sub("ا", text)
    text = ARABIC_YA_VARIATIONS.sub("ي", text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = ARABIC_TATWEEL.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_arabic(text: str) -> bool:
    """Check if text is primarily Arabic (>30% Arabic characters)."""
    if not text:
        return False
    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    total_chars = len(re.findall(r"[\w]", text))
    if total_chars == 0:
        return False
    return arabic_chars / total_chars > 0.3
