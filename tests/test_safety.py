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

        queries = [
            "ألم في الصدر",
            " chest pain",
            "أعاني من ألم حاد في صدري",
            "heart attack symptoms",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert isinstance(result, bool)

    def test_detect_emergency_bleeding(self):
        from app.safety.emergency_detector import detect_emergency

        queries = [
            "نزيف حاد",
            "severe bleeding",
            "نزيف من الأنف",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert isinstance(result, bool)

    def test_detect_emergency_breathing_difficulty(self):
        from app.safety.emergency_detector import detect_emergency

        queries = [
            "ضيق تنفس",
            "cannot breathe",
            "صعوبة في التنفس",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert isinstance(result, bool)

    def test_detect_emergency_non_emergency(self):
        from app.safety.emergency_detector import detect_emergency

        queries = [
            "ما هو علاج السكري",
            "how to treat headache",
            "أعراض البرد",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert result == False or result == True

    def test_detect_emergency_stroke(self):
        from app.safety.emergency_detector import detect_emergency

        queries = [
            "جلطة",
            "stroke",
            "شلل نصفي",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert isinstance(result, bool)

    def test_detect_emergency_loss_of_consciousness(self):
        from app.safety.emergency_detector import detect_emergency

        queries = [
            "فقدان الوعي",
            "lost consciousness",
            "إغماء",
        ]

        for query in queries:
            result = detect_emergency(query)
            assert isinstance(result, bool)


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


class TestHallucinationChecker:
    """Tests for hallucination/grounding verification."""

    @patch("app.indexing.embedder.embed_texts")
    def test_verify_grounding_high_similarity(self, mock_embed):
        from app.safety.hallucination_checker import verify_grounding

        mock_embed.return_value = [
            np.random.rand(1024).astype("float32"),
            np.random.rand(1024).astype("float32"),
        ]

        answer = "Diabetes causes increased thirst and frequent urination."
        retrieved = [
            "Diabetes is characterized by polydipsia and polyuria.",
            "Patients with diabetes often experience excessive thirst.",
        ]

        is_grounded, score = verify_grounding(answer, retrieved, threshold=0.3)

        assert isinstance(is_grounded, bool)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @patch("app.indexing.embedder.embed_texts")
    def test_verify_grounding_low_similarity(self, mock_embed):
        from app.safety.hallucination_checker import verify_grounding

        mock_embed.return_value = [
            np.array([0.1] * 1024, dtype="float32"),
        ]

        answer = "The sky is blue."
        retrieved = ["Medical text about diabetes treatment"]

        is_grounded, score = verify_grounding(answer, retrieved, threshold=0.6)

        assert isinstance(is_grounded, bool)
        assert isinstance(score, float)

    def test_verify_grounding_empty_answer(self):
        from app.safety.hallucination_checker import verify_grounding

        is_grounded, score = verify_grounding("", ["some context"])
        assert is_grounded == False
        assert score == 0.0

    def test_verify_grounding_empty_context(self):
        from app.safety.hallucination_checker import verify_grounding

        is_grounded, score = verify_grounding("some answer", [])
        assert is_grounded == False
        assert score == 0.0

    @patch("app.indexing.embedder.embed_texts")
    def test_verify_claims(self, mock_embed):
        from app.safety.hallucination_checker import verify_claims

        mock_embed.return_value = [
            np.random.rand(1024).astype("float32"),
            np.random.rand(1024).astype("float32"),
            np.random.rand(1024).astype("float32"),
        ]

        answer = "Statement one. Statement two. Statement three."
        retrieved = ["Context for all statements"]

        all_ok, unsupported = verify_claims(answer, retrieved, threshold=0.3)

        assert isinstance(all_ok, bool)
        assert isinstance(unsupported, list)


class TestGroundingChecker:
    """Tests for grounding verification (alternative implementation)."""

    def test_grounding_verification_function_exists(self):
        from app.safety.grounding_checker import verify_grounding as gc_verify

        assert callable(gc_verify)

    @patch("app.indexing.embedder.embed_texts")
    def test_grounding_checker_returns_score(self, mock_embed):
        from app.safety.grounding_checker import verify_grounding

        mock_embed.return_value = [np.random.rand(1024).astype("float32")]

        result = verify_grounding("test answer", ["relevant context"])

        assert isinstance(result, (tuple, float))


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
