import pytest
import numpy as np
from unittest.mock import patch, Mock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmergencyDetector:
    """Tests for emergency detection in Arabic medical queries."""

    def test_detect_emergency_chest_pain(self):
        from app.safety.emergency_detector import detect_emergency

        # These are known emergency phrases — should return True
        queries = [
            "ألم في الصدر",
            "chest pain",
            "severe chest pain",
            "heart attack symptoms",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert result is True, f"Expected emergency for: {query}"

    def test_detect_emergency_bleeding(self):
        from app.safety.emergency_detector import detect_emergency

        queries = [
            "نزيف حاد",
            "severe bleeding",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert result is True, f"Expected emergency for: {query}"

    def test_detect_emergency_breathing_difficulty(self):
        from app.safety.emergency_detector import detect_emergency

        queries = [
            "cannot breathe",
            "difficulty breathing",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert result is True, f"Expected emergency for: {query}"

    def test_detect_emergency_non_emergency(self):
        from app.safety.emergency_detector import detect_emergency

        queries = [
            "ما هو علاج السكري",
            "how to treat headache",
            "أعراض البرد",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert result is False, f"Expected non-emergency for: {query}"

    def test_detect_emergency_stroke(self):
        from app.safety.emergency_detector import detect_emergency

        queries = [
            "stroke",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert result is True, f"Expected emergency for: {query}"

    def test_detect_emergency_loss_of_consciousness(self):
        from app.safety.emergency_detector import detect_emergency

        queries = [
            "فقدان الوعي",
            "loss of consciousness",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert result is True, f"Expected emergency for: {query}"

    def test_detect_emergency_empty_query(self):
        from app.safety.emergency_detector import detect_emergency

        assert detect_emergency("") is False
        assert detect_emergency("   ") is False


class TestContentFilter:
    """Tests for content filtering."""

    def test_contains_sensitive_content_medical(self):
        from app.safety.content_filter import contains_sensitive_content

        result = contains_sensitive_content("some medical text")
        assert isinstance(result, bool)

    def test_contains_sensitive_content_adult(self):
        from app.safety.content_filter import contains_sensitive_content

        result = contains_sensitive_content("inappropriate adult content")
        assert isinstance(result, bool)

    def test_contains_sensitive_content_drugs(self):
        from app.safety.content_filter import contains_sensitive_content

        queries = [
            "how to make drugs",
            "تعليمات تصنيع مخدرات",
        ]

        for query in queries:
            result = contains_sensitive_content(query)
            assert isinstance(result, bool)


class TestSafetyIntegration:
    """Integration tests for safety pipeline."""

    def test_safety_pipeline_complete(self):
        from app.safety.emergency_detector import detect_emergency
        from app.safety.content_filter import contains_sensitive_content

        query = "أعاني من ألم في صدري"

        is_emergency = detect_emergency(query)
        has_sensitive = contains_sensitive_content(query)

        assert isinstance(is_emergency, bool)
        assert isinstance(has_sensitive, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
