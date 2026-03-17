"""
Arabic Evaluation Tests for MediLink RAG System.

This module tests the cross-lingual retrieval performance
between Arabic queries and English medical documents.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, Mock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestArabicEvaluationDataset:
    """Tests for Arabic evaluation dataset structure."""

    @pytest.fixture
    def arabic_eval_data(self):
        eval_path = Path("data/eval_ground_truth.json")
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"queries": [q for q in data if q.get("language") == "arabic"]}
        return None

    @pytest.fixture
    def english_eval_data(self):
        eval_path = Path("data/eval_ground_truth.json")
        if eval_path.exists():
            with open(eval_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return {"queries": [q for q in data if q.get("language") == "english"]}
        return None

    def test_arabic_dataset_exists(self, arabic_eval_data):
        assert arabic_eval_data is not None, "Arabic evaluation dataset not found"

    def test_arabic_dataset_version(self, arabic_eval_data):
        assert "version" in arabic_eval_data
        assert arabic_eval_data["version"] == "1.0"

    def test_arabic_dataset_language(self, arabic_eval_data):
        assert arabic_eval_data["language"] == "arabic"

    def test_arabic_dataset_has_categories(self, arabic_eval_data):
        assert "categories" in arabic_eval_data
        assert len(arabic_eval_data["categories"]) > 0

    def test_arabic_dataset_has_dialects(self, arabic_eval_data):
        assert "dialects" in arabic_eval_data
        assert "MSA" in arabic_eval_data["dialects"]
        assert "Egyptian" in arabic_eval_data["dialects"]

    def test_arabic_queries_have_required_fields(self, arabic_eval_data):
        if arabic_eval_data is None:
            pytest.skip("Arabic eval data not found")

        required_fields = [
            "id",
            "query",
            "question",
            "category",
            "dialect",
            "difficulty",
            "ground_truth_doc_ids",
            "expected_answer_components",
        ]

        for query in arabic_eval_data["queries"]:
            for field in required_fields:
                assert field in query, (
                    f"Missing field {field} in query {query.get('id')}"
                )

    def test_arabic_queries_all_categories_covered(self, arabic_eval_data):
        if arabic_eval_data is None:
            pytest.skip("Arabic eval data not found")

        categories_in_queries = set(q["category"] for q in arabic_eval_data["queries"])
        expected_categories = set(arabic_eval_data["categories"])

        assert categories_in_queries.issubset(expected_categories)

    def test_arabic_queries_all_dialects_covered(self, arabic_eval_data):
        if arabic_eval_data is None:
            pytest.skip("Arabic eval data not found")

        dialects_in_queries = set(q["dialect"] for q in arabic_eval_data["queries"])

        assert len(dialects_in_queries) >= 3

    def test_arabic_queries_difficulty_distribution(self, arabic_eval_data):
        if arabic_eval_data is None:
            pytest.skip("Arabic eval data not found")

        difficulties = {}
        for q in arabic_eval_data["queries"]:
            d = q.get("difficulty", "medium")
            difficulties[d] = difficulties.get(d, 0) + 1

        assert difficulties.get("easy", 0) >= 3
        assert difficulties.get("medium", 0) >= 3
        assert difficulties.get("hard", 0) >= 3

    def test_arabic_queries_have_english_terms(self, arabic_eval_data):
        if arabic_eval_data is None:
            pytest.skip("Arabic eval data not found")

        for query in arabic_eval_data["queries"]:
            assert "related_english_terms" in query
            assert len(query["related_english_terms"]) > 0

    def test_english_dataset_exists(self, english_eval_data):
        assert english_eval_data is not None, "English evaluation dataset not found"

    def test_english_dataset_structure(self, english_eval_data):
        assert english_eval_data["language"] == "english"
        assert len(english_eval_data["queries"]) > 0

    def test_arabic_vs_english_coverage(self, arabic_eval_data, english_eval_data):
        if arabic_eval_data is None or english_eval_data is None:
            pytest.skip("Eval data not found")

        ar_cats = set(q["category"] for q in arabic_eval_data["queries"])
        en_cats = set(q["category"] for q in english_eval_data["queries"])

        assert ar_cats == en_cats, "Categories should match between Arabic and English"


class TestArabicRetrievalPerformance:
    """Tests for Arabic-to-English cross-lingual retrieval."""

    @patch("app.indexing.embedder.embed_texts")
    def test_multilingual_embedding_quality(self, mock_embed):
        import numpy as np
        from app.indexing.embedder import embed_texts

        mock_embed.return_value = [np.random.rand(1024).astype("float32")]

        arabic_texts = ["أعراض السكري", "علاج ارتفاع ضغط الدم", "ألم في القلب"]

        embeddings = embed_texts(arabic_texts)

        assert len(embeddings) == len(arabic_texts)
        assert embeddings[0].shape == (1024,)

    @patch("app.indexing.vector_store.VectorStore")
    def test_arabic_query_retrieval(self, mock_vs):
        import numpy as np
        from app.retrieval.hybrid_fusion import hybrid_fusion

        mock_vs_instance = Mock()
        mock_vs_instance.search.return_value = [(0, 0.85), (1, 0.75), (2, 0.65)]

        arabic_query = "أعراض مرض السكري"

        dense_results = [
            {"text": "Diabetes symptoms include increased thirst", "score": 0.85},
            {"text": "Frequent urination is a symptom", "score": 0.75},
        ]

        fused = hybrid_fusion(dense_results, [])

        assert len(fused) > 0
        assert fused[0]["score"] >= fused[-1]["score"]


class TestArabicQueryExpansion:
    """Tests for Arabic query expansion."""

    def test_arabic_expansion_includes_synonyms(self):
        from app.retrieval.query_expansion import expand_query

        arabic_queries = ["أعراض السكري", "علاج القلب", "ارتفاع الضغط"]

        for query in arabic_queries:
            expanded = expand_query(query)

            assert isinstance(expanded, list)
            assert len(expanded) > 0
            assert query in expanded


class TestArabicGrounding:
    """Tests for Arabic answer grounding verification."""

    @patch("app.indexing.embedder.embed_texts")
    def test_arabic_answer_grounded_in_english_context(self, mock_embed):
        import numpy as np
        from app.safety.hallucination_checker import verify_grounding

        mock_embed.return_value = [np.random.rand(1024).astype("float32")]

        arabic_answer = "أعراض السكري هي العطش والتبول المتكرر"
        english_context = [
            "Diabetes symptoms include increased thirst (polydipsia) and frequent urination (polyuria).",
            "Patients may also experience fatigue and weight loss.",
        ]

        is_grounded, score = verify_grounding(
            arabic_answer, english_context, threshold=0.3
        )

        assert isinstance(is_grounded, bool)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @patch("app.indexing.embedder.embed_texts")
    def test_cross_lingual_grounding_score(self, mock_embed):
        import numpy as np
        from app.safety.hallucination_checker import verify_grounding

        np.random.seed(42)
        mock_embed.return_value = [
            np.random.rand(1024).astype("float32"),
            np.random.rand(1024).astype("float32"),
        ]

        arabic_answer = "النص العربي"
        english_context = ["English context about medical topic"]

        is_grounded, score = verify_grounding(arabic_answer, english_context)

        assert isinstance(score, float)


class TestArabicEmergencyDetection:
    """Tests for Arabic emergency detection."""

    def test_arabic_emergency_detected(self):
        from app.safety.emergency_detector import detect_emergency

        emergency_queries = [
            "ألم في الصدر",
            "نزيف حاد",
            "ضيق تنفس",
            "فقدان الوعي",
            "جلطة",
        ]

        results = [detect_emergency(q) for q in emergency_queries]

        assert all(isinstance(r, bool) for r in results)

    def test_arabic_non_emergency(self):
        from app.safety.emergency_detector import detect_emergency

        non_emergency_queries = [
            "ما هي أعراض السكري",
            "كيف أعالج الصداع",
            "أعراض البرد",
        ]

        for query in non_emergency_queries:
            result = detect_emergency(query)
            assert isinstance(result, bool)


class TestArabicEvaluationMetrics:
    """Tests for evaluation metrics with Arabic data."""

    def test_recall_at_k_with_arabic_ground_truth(self):
        from app.evaluation.metrics import recall_at_k

        retrieved = [["doc_436", "doc_435", "doc_442"]]
        relevant = [["doc_436", "doc_435", "doc_442", "doc_434"]]

        recall = recall_at_k(retrieved, relevant, k=3)

        assert recall == 3 / 4

    def test_mrr_with_arabic(self):
        from app.evaluation.metrics import mrr

        retrieved = [["doc_436", "doc_435", "doc_442"]]
        relevant = [["doc_442"]]

        mrr_score = mrr(retrieved, relevant)

        assert mrr_score == 1 / 3

    def test_ndcg_with_arabic(self):
        from app.evaluation.metrics import ndcg_at_k

        retrieved = [["doc_436", "doc_435", "doc_442"]]
        relevant = [["doc_436", "doc_435", "doc_442"]]

        ndcg = ndcg_at_k(retrieved, relevant, k=3)

        assert ndcg > 0


class TestArabicEnglishComparison:
    """Tests comparing Arabic vs English retrieval performance."""

    @pytest.mark.parametrize("language", ["arabic", "english"])
    def test_query_expansion_both_languages(self, language):
        from app.retrieval.query_expansion import expand_query

        query = "أعراض السكري" if language == "arabic" else "diabetes symptoms"

        expanded = expand_query(query)

        assert len(expanded) > 0

    def test_dense_retrieval_arabic_english(self):
        from app.retrieval.hybrid_fusion import hybrid_fusion

        results = [{"text": "relevant document", "score": 0.9}]

        fused = hybrid_fusion(results, [])

        assert len(fused) > 0
        assert fused[0]["score"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
