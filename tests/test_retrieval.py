import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHybridFusion:
    """Tests for hybrid fusion retrieval pipeline."""

    def test_reciprocal_rank_fusion_empty_lists(self):
        from app.retrieval.hybrid_fusion import reciprocal_rank_fusion

        result = reciprocal_rank_fusion([])
        assert result == []

    def test_reciprocal_rank_fusion_single_list(self):
        from app.retrieval.hybrid_fusion import reciprocal_rank_fusion

        results = [[{"text": "doc1", "score": 1.0}, {"text": "doc2", "score": 0.9}]]
        fused = reciprocal_rank_fusion(results, k=60)
        assert len(fused) == 2
        assert fused[0]["text"] == "doc1"
        assert fused[0]["score"] > fused[1]["score"]

    def test_reciprocal_rank_fusion_duplicate_docs(self):
        from app.retrieval.hybrid_fusion import reciprocal_rank_fusion

        list1 = [{"text": "doc1", "score": 1.0}]
        list2 = [{"text": "doc1", "score": 0.9}]
        fused = reciprocal_rank_fusion([list1, list2], k=60)
        assert len(fused) == 1
        assert fused[0]["score"] > 0

    def test_hybrid_fusion_dense_only(self):
        from app.retrieval.hybrid_fusion import hybrid_fusion

        dense = [{"text": "doc1", "score": 1.0}, {"text": "doc2", "score": 0.8}]
        result = hybrid_fusion(dense, [])
        assert len(result) == 2
        assert "dense_score" in result[0]
        assert "bm25_score" in result[0]

    def test_hybrid_fusion_both_empty(self):
        from app.retrieval.hybrid_fusion import hybrid_fusion

        result = hybrid_fusion([], [])
        assert result == []


class TestReranker:
    """Tests for reranking functionality."""

    @patch("app.retrieval.reranker.default_reranker")
    def test_rerank_returns_list(self, mock_reranker):
        from app.retrieval.reranker import rerank

        mock_reranker.rerank.return_value = [0.9, 0.8, 0.7]

        docs = [{"text": "doc1"}, {"text": "doc2"}, {"text": "doc3"}]
        result = rerank("test query", docs, top_k=3)

        assert isinstance(result, list)
        assert len(result) <= 3

    def test_rerank_with_empty_docs(self):
        from app.retrieval.reranker import rerank

        result = rerank("test query", [], top_k=10)
        assert result == []


class TestQueryExpansion:
    """Tests for query expansion."""

    def test_expand_query_returns_list(self):
        from app.retrieval.query_expansion import expand_query

        result = expand_query("diabetes symptoms")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_expand_query_arabic(self):
        from app.retrieval.query_expansion import expand_query

        result = expand_query("أعراض السكري")
        assert isinstance(result, list)
        assert len(result) > 0


class TestMetadataFilter:
    """Tests for metadata filtering."""

    def test_filter_by_metadata_specialty(self):
        from app.retrieval.metadata_filter import filter_by_metadata

        docs = [
            {"text": "cardiology info", "metadata": {"specialty": "cardiology"}},
            {"text": "general info", "metadata": {"specialty": "general"}},
            {"text": "cardio info 2", "metadata": {"specialty": "cardiology"}},
        ]

        result = filter_by_metadata(docs, specialty="cardiology")
        assert len(result) == 2

    def test_filter_by_metadata_no_match(self):
        from app.retrieval.metadata_filter import filter_by_metadata

        docs = [
            {"text": "cardiology info", "metadata": {"specialty": "cardiology"}},
        ]

        result = filter_by_metadata(docs, specialty="neurology")
        assert len(result) == 0

    def test_filter_by_metadata_empty_docs(self):
        from app.retrieval.metadata_filter import filter_by_metadata

        result = filter_by_metadata([], specialty="cardiology")
        assert result == []


class TestContextCompressor:
    """Tests for context compression."""

    def test_compress_context_returns_list(self):
        from app.retrieval.context_compressor import compress_context

        docs = [
            {"text": "This is a very long text " * 50},
            {"text": "Another long text " * 50},
        ]

        result = compress_context("test query", docs, max_tokens=100)
        assert isinstance(result, list)

    def test_compress_context_empty(self):
        from app.retrieval.context_compressor import compress_context

        result = compress_context("test", [], max_tokens=100)
        assert isinstance(result, list)


class TestRetrievalPipeline:
    """Integration tests for full retrieval pipeline."""

    @patch("app.indexing.embedder.embed_texts")
    @patch("app.indexing.vector_store.VectorStore")
    def test_retrieval_returns_formatted_results(self, mock_vs, mock_embed):
        from app.retrieval.hybrid_fusion import hybrid_fusion

        mock_embed.return_value = [np.random.rand(1024).astype("float32")]

        mock_vs_instance = Mock()
        mock_vs_instance.search.return_value = [(0, 0.9), (1, 0.8)]

        docs = [
            {"text": "test doc 1", "score": 0.9, "metadata": {"page": 1}},
            {"text": "test doc 2", "score": 0.8, "metadata": {"page": 2}},
        ]

        fused = hybrid_fusion(docs, [])
        assert len(fused) > 0
        assert "score" in fused[0]


class TestRetrievalMetrics:
    """Tests for retrieval metric calculations."""

    def test_recall_at_k_perfect(self):
        from app.evaluation.metrics import recall_at_k

        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [["doc1", "doc2", "doc3"]]

        recall = recall_at_k(retrieved, relevant, k=3)
        assert recall == 1.0

    def test_recall_at_k_partial(self):
        from app.evaluation.metrics import recall_at_k

        retrieved = [["doc1", "doc2"]]
        relevant = [["doc1", "doc2", "doc3"]]

        recall = recall_at_k(retrieved, relevant, k=2)
        assert recall == 2 / 3

    def test_precision_at_k_perfect(self):
        from app.evaluation.metrics import precision_at_k

        retrieved = [["doc1", "doc2"]]
        relevant = [["doc1", "doc2"]]

        precision = precision_at_k(retrieved, relevant, k=2)
        assert precision == 1.0

    def test_mrr_perfect(self):
        from app.evaluation.metrics import mrr

        retrieved = [["doc1", "doc2", "doc3"]]
        relevant = [["doc1"]]

        mrr_score = mrr(retrieved, relevant)
        assert mrr_score == 1.0

    def test_mrr_no_hit(self):
        from app.evaluation.metrics import mrr

        retrieved = [["doc1", "doc2"]]
        relevant = [["doc3"]]

        mrr_score = mrr(retrieved, relevant)
        assert mrr_score == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
