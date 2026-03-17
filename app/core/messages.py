"""Externalized messages for the MediLink API."""

from typing import Dict


class Messages:
    """Centralized message strings for the API."""

    EMERGENCY_ESCALATION: str = "هذه حالة طبية طارئة. يرجى الاتصال بالإسعاف فوراً."

    NO_RETRIEVAL: str = "لم يتم العثور على معلومات كافية للإجابة."

    NO_RESULTS: str = "عذراً، لم أتمكن من العثور على معلومات ذات صلة باستعلامك."

    ERROR_PROCESSING: str = "عذراً، حدث خطأ أثناء معالجة طلبك."

    LOW_CONFIDENCE: str = "لم أتمكن من الإجابة على استعلامك بدقة عالية."

    CONTENT_FILTERED: str = "عذراً، لا يمكنني معالجة هذا الطلب."

    INVALID_QUERY: str = "يرجى تقديم استعلام صحيح."

    @classmethod
    def get_message(cls, key: str, default: str = "") -> str:
        """Get a message by key."""
        return getattr(cls, key.upper(), default)


MESSAGES = Messages()
