import logging
import re
from typing import Union, List, Optional, Callable

logger = logging.getLogger(__name__)

ARABIC_ALEF_VARIATIONS = re.compile("[إأآا]")
ARABIC_YA_VARIATIONS = re.compile("[ىي]")
ARABIC_TA_MARBUTA = re.compile("[ة]")
ARABIC_DIACRITICS = re.compile(r"[\u064B-\u0652]")
ARABIC_TATWEEL = re.compile("ـ")
PUNCTUATION_PATTERN = re.compile(r"[^\w\s\u0600-\u06FF\u0750-\u077F]")


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text by standardizing character variations.

    Handles:
    - Alef variations (إأآا → ا)
    - Ya variations (ىي → ي)
    - Ta marbuta to Ha (ة → ه)
    - Diacritics removal
    - Tatweel (ـ) removal
    """
    if not isinstance(text, str):
        text = str(text)

    text = ARABIC_ALEF_VARIATIONS.sub("ا", text)
    text = ARABIC_YA_VARIATIONS.sub("ي", text)
    text = ARABIC_TA_MARBUTA.sub("ه", text)
    text = ARABIC_DIACRITICS.sub("", text)
    text = ARABIC_TATWEEL.sub("", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


def remove_punctuation(text: str, keep_arabic: bool = True) -> str:
    """Remove unnecessary punctuation while preserving Arabic and English."""
    if not isinstance(text, str):
        return ""

    if keep_arabic:
        text = PUNCTUATION_PATTERN.sub(" ", text)
    else:
        text = re.sub(r"[^\w\s]", " ", text)

    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_query(text: str) -> str:
    """Complete query preprocessing pipeline for retrieval.

    Applies:
    - Arabic normalization
    - Punctuation removal
    - Whitespace normalization

    Args:
        text: Input query string

    Returns:
        Preprocessed query string
    """
    if not text:
        return ""

    text = normalize_arabic(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)

    return text


def preprocess_document(text: str) -> str:
    """Preprocessing for documents (less aggressive than queries).

    Preserves more structure but normalizes Arabic.
    """
    if not text:
        return ""

    text = normalize_arabic(text)
    text = normalize_whitespace(text)

    return text


class TextPreprocessor:
    """Preprocessor with configurable pipeline."""

    def __init__(
        self,
        normalize_arabic: bool = True,
        remove_punctuation: bool = True,
        lowercase: bool = True,
    ) -> None:
        self.normalize_arabic = normalize_arabic
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase

    def process(self, text: str) -> str:
        """Process text through the configured pipeline."""
        if not text:
            return ""

        if self.normalize_arabic:
            text = normalize_arabic(text)

        if self.remove_punctuation:
            text = remove_punctuation(text)

        if self.lowercase:
            text = text.lower()

        text = normalize_whitespace(text)

        return text

    def __call__(self, text: str) -> str:
        return self.process(text)


def clean_text(text: str) -> str:
    """Complete text cleaning pipeline.

    Deprecated: Use preprocess_query or preprocess_document instead.
    """
    if not isinstance(text, str):
        return ""

    text = normalize_arabic(text)
    text = remove_punctuation(text)
    text = normalize_whitespace(text)

    return text
