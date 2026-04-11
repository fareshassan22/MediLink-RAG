import logging
import re
from typing import List, Optional, Set
from pathlib import Path

from app.utils.arabic import normalize_arabic, is_arabic  # noqa: F401

logger = logging.getLogger(__name__)
ARABIC_STOPWORDS: Set[str] = {
    "في",
    "من",
    "على",
    "إلى",
    "عن",
    "مع",
    "هذا",
    "هذه",
    "ذلك",
    "تلك",
    "التي",
    "الذي",
    "الذي",
    "التى",
    "اللذين",
    "اللتين",
    "اللواتي",
    "هو",
    "هي",
    "هم",
    "هن",
    "أنا",
    "نحن",
    "أنت",
    "أنتم",
    "أنتن",
    "كان",
    "كانت",
    "كانوا",
    "يكون",
    "تكون",
    "لم",
    "لا",
    "ما",
    "قد",
    "كل",
    "بعض",
    "أي",
    "أحد",
    "اثنان",
    "ثلاثة",
    "أربعة",
    "خمسة",
    "و",
    "أو",
    "ثم",
    "ف",
    "ب",
    "ل",
    "ك",
    "ف",
    "ع",
    "ت",
    "بعض",
    "بين",
    "حتى",
    "منذ",
    "خلال",
    "بدون",
    "فقط",
    "أيضاً",
    "جداً",
    "إذا",
    "لو",
    "لما",
    "حيث",
    "كيف",
    "متى",
    "لماذا",
    "هل",
    "أ",
    "إن",
    "أن",
    "كان",
    "ليست",
    "ليس",
    "ما",
    "لا",
    "لن",
    "لم",
    "ألم",
    "أليس",
    "بلى",
    "جرى",
    "مثل",
    "كأن",
    "لكن",
    "بل",
    "حتى",
    "لو",
    "لولا",
    "لوما",
    "ألا",
    "أما",
    "أوان",
    "آ",
}

STEMMING_SUFFIXES: List[str] = [
    "ون",
    "ين",
    "ات",
    "اتك",
    "اتكم",
    "اته",
    "اتي",
    "وك",
    "ة",
    "تين",
    "تينك",
    "تينكم",
    "ته",
    "تي",
    "ينك",
    "ينكم",
    "هم",
    "هن",
    "نا",
    "ني",
    "كم",
    "كن",
    "ه",
    "ك",
    "ي",
    "ت",
]

PREFIXES_3: List[str] = [
    "فال",
    "فت",
    "فس",
    "فك",
    "فل",
    "فن",
    "في",
    "ال",
    "ال",
    "ال",
    "وال",
    "ولات",
    "ولات",
    "ولات",
    "بال",
    "بالت",
    "بالس",
    "بالك",
    "بالل",
    "بالن",
    "بي",
    "لل",
    "للت",
    "للس",
    "للك",
    "للل",
    "للن",
    "لي",
    "كال",
    "كالت",
    "كالس",
    "كالك",
    "كالل",
    "كالن",
    "كي",
    "سأ",
    "سي",
    "سن",
    "ست",
    "سو",
    "ستم",
    "ستن",
    "أف",
    "أف",
    "أس",
    "أس",
    "أك",
    "أل",
    "أن",
    "أو",
    "نأ",
    "نص",
    "نط",
    "نف",
    "نم",
    "نه",
    "نو",
]

PREFIXES_2: List[str] = [
    "ف",
    "ب",
    "ل",
    "ك",
    "ع",
    "ت",
    "ن",
    "أ",
    "ي",
    "و",
    "في",
    "بي",
    "لي",
    "كي",
    "تي",
    "ني",
    "وي",
    "أي",
    "أف",
    "أس",
    "أك",
    "أل",
    "أن",
    "أو",
    "سن",
    "ست",
    "سو",
    "سي",
    "سأ",
    "لن",
    "لم",
    "لا",
    "لو",
    "ال",
]


def simple_arabic_stem(word: str) -> str:
    """Simple Arabic stemmer - removes common suffixes and prefixes."""
    if not word or len(word) < 3:
        return word

    word = word.strip()

    word = re.sub(r"^(ال)", "", word)

    for suffix in sorted(STEMMING_SUFFIXES, key=len, reverse=True):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break

    for prefix in sorted(PREFIXES_3, key=len, reverse=True):
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            word = word[len(prefix) :]
            break

    for prefix in sorted(PREFIXES_2, key=len, reverse=True):
        if word.startswith(prefix) and len(word) - len(prefix) >= 3:
            word = word[len(prefix) :]
            break

    return word


def tokenize_arabic(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize Arabic text with proper stemming and stopword removal."""
    text = normalize_arabic(text)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    words = text.split()

    tokens = []
    for word in words:
        word = word.strip()
        if not word:
            continue
        if len(word) < 2:
            continue
        stemmed = simple_arabic_stem(word)
        if remove_stopwords and stemmed.lower() in ARABIC_STOPWORDS:
            continue
        tokens.append(stemmed)

    return tokens


def tokenize_english(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize English text with simple normalization."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()

    if remove_stopwords:
        en_stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "as",
            "is",
            "was",
            "are",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "shall",
            "can",
            "need",
            "it",
            "its",
            "this",
            "that",
            "these",
            "those",
            "i",
            "you",
            "he",
            "she",
            "we",
            "they",
            "what",
            "which",
            "who",
            "whom",
            "how",
            "when",
            "where",
            "why",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "just",
            "also",
        }
        words = [w for w in words if w not in en_stopwords]

    return [w for w in words if len(w) >= 2]


def tokenize_bilingual(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize mixed Arabic/English text."""
    text = normalize_arabic(text)

    arabic_tokens = []
    english_tokens = []

    segments = re.split(r"([\u0600-\u06FF]+)", text)

    for segment in segments:
        if not segment:
            continue
        if is_arabic(segment):
            arabic_tokens.extend(tokenize_arabic(segment, remove_stopwords))
        else:
            english_tokens.extend(tokenize_english(segment, remove_stopwords))

    return arabic_tokens + english_tokens


def preprocess_query(text: str) -> str:
    """Complete query preprocessing pipeline for retrieval."""
    if not text:
        return ""
    return normalize_arabic(text)


def preprocess_document(text: str) -> str:
    """Preprocessing for documents."""
    if not text:
        return ""
    return normalize_arabic(text)
