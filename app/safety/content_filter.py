import re

# ---------- Keyword-based sensitive terms ----------
SENSITIVE_TERMS = [
    "اسم المريض",
    "رقم الهوية",
    "العنوان",
    "phone number",
    "ssn",
    "social security",
    "credit card",
    "رقم الهاتف",
    "رقم البطاقة",
    "رقم الجواز",
    "عنوان المنزل",
    "كلمة المرور",
    "password",
]

# ---------- PII regex patterns ----------
_PII_PATTERNS = [
    re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),          # SSN-like
    re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), # Credit card
    re.compile(r"\b[\w.-]+@[\w.-]+\.\w{2,}\b"),                 # Email
    re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),  # Phone
    re.compile(r"\b\d{9,10}\b"),                                 # National ID
]


def contains_sensitive_content(text: str) -> bool:
    """Check text for PII or sensitive terms."""
    if not text:
        return False
    text_lower = text.lower()
    if any(term in text_lower for term in SENSITIVE_TERMS):
        return True
    return any(pat.search(text) for pat in _PII_PATTERNS)