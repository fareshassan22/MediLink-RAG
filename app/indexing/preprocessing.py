import logging
import re
from typing import Union, List, Optional, Callable

from app.utils.arabic import normalize_arabic  # noqa: F401 – re-exported

logger = logging.getLogger(__name__)

PUNCTUATION_PATTERN = re.compile(r"[^\w\s\u0600-\u06FF\u0750-\u077F]")


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



