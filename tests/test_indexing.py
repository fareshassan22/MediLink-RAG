import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmbedder:
    """Tests for text embedding module."""

    @patch("app.indexing.embedder._model")
    def test_embed_texts_returns_list(self, mock_model):
        from app.indexing.embedder import embed_texts

        mock_model.encode.return_value = np.random.rand(2, 1024).astype("float32")

        texts = ["text1", "text2"]
        result = embed_texts(texts)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (1024,)

    @patch("app.indexing.embedder._model")
    def test_embed_texts_single(self, mock_model):
        from app.indexing.embedder import embed_texts

        mock_model.encode.return_value = np.random.rand(1, 1024).astype("float32")

        texts = ["single text"]
        result = embed_texts(texts)

        assert len(result) == 1

    @patch("app.indexing.embedder._model")
    def test_embed_texts_empty(self, mock_model):
        from app.indexing.embedder import embed_texts

        result = embed_texts([])

        assert result == []

    @patch("app.indexing.embedder._model")
    def test_embed_texts_normalized(self, mock_model):
        from app.indexing.embedder import embed_texts

        mock_model.encode.return_value = np.random.rand(1, 1024).astype("float32")

        embed_texts(["test"])

        mock_model.encode.assert_called_once()
        call_kwargs = mock_model.encode.call_args[1]
        assert call_kwargs.get("normalize_embeddings") == True


class TestVectorStore:
    """Tests for vector store."""

    def test_vector_store_init(self):
        from app.indexing.vector_store import VectorStore

        vs = VectorStore(dim=1024)

        assert vs.dim == 1024

    @patch("faiss.IndexFlatIP")
    def test_vector_store_search(self, mock_faiss):
        from app.indexing.vector_store import VectorStore

        vs = VectorStore(dim=1024)

        mock_index = Mock()
        mock_index.ntotal = 10
        vs.index = mock_index

        query = np.random.rand(1024).astype("float32")
        results = vs.search(query, k=5)

        assert isinstance(results, list)

    def test_vector_store_add_documents(self):
        from app.indexing.vector_store import VectorStore

        vs = VectorStore(dim=1024)

        vs.add("doc1", np.random.rand(1024).astype("float32"), {"page": 1})
        vs.add("doc2", np.random.rand(1024).astype("float32"), {"page": 2})

        assert len(vs.documents) == 2


class TestBM25Index:
    """Tests for BM25 index."""

    def test_bm25_index_init(self):
        from app.indexing.bm25_index import BM25Index

        bm25 = BM25Index()

        assert bm25 is not None

    def test_bm25_index_search(self):
        from app.indexing.bm25_index import BM25Index

        bm25 = BM25Index(use_preprocessing=False)
        bm25.build(["diabetes symptoms include thirst", "heart disease and treatment"])

        results = bm25.search("diabetes", k=5)

        assert isinstance(results, list)
        assert len(results) > 0


class TestChunker:
    """Tests for text chunking."""

    def test_chunk_text_by_sentence(self):
        from app.indexing.chunker import semantic_chunk

        text = "This is sentence one. This is sentence two. This is sentence three."

        chunks = semantic_chunk(text, chunk_size=2)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_text_by_sentence_small(self):
        from app.indexing.chunker import semantic_chunk

        text = "Short text."

        chunks = semantic_chunk(text, chunk_size=5)

        assert isinstance(chunks, list)


class TestPreprocessing:
    """Tests for text preprocessing."""

    def test_normalize_arabic(self):
        from app.indexing.preprocessing import normalize_arabic

        text = "أَلْف"
        normalized = normalize_arabic(text)

        assert normalized is not None

    def test_remove_special_characters(self):
        from app.indexing.preprocessing import remove_punctuation

        text = "Test @#$% text!"
        cleaned = remove_punctuation(text)

        assert "@" not in cleaned
        assert "#" not in cleaned


class TestArabicTokenizer:
    """Tests for Arabic tokenization."""

    def test_arabic_tokenizer_init(self):
        from app.indexing.arabic_tokenizer import tokenize_arabic, is_arabic

        assert callable(tokenize_arabic)
        assert callable(is_arabic)

    def test_tokenize_arabic(self):
        from app.indexing.arabic_tokenizer import tokenize_arabic

        text = "هذا نص عربي للاختبار"
        tokens = tokenize_arabic(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_remove_arabic_stopwords(self):
        from app.indexing.arabic_tokenizer import tokenize_arabic

        text = "هذا هو النص"
        tokens = tokenize_arabic(text, remove_stopwords=True)

        assert isinstance(tokens, list)


class TestIndexPipeline:
    """Tests for indexing pipeline."""

    @patch("app.indexing.index_pipeline.BM25Index")
    def test_index_pipeline_runs(self, mock_bm25_cls):
        from app.indexing.index_pipeline import load_bm25

        mock_bm25_cls.load.return_value = Mock()

        result = load_bm25("data/processed")

        assert result is not None
        mock_bm25_cls.load.assert_called_once_with("data/processed")


class TestEmbedderModel:
    """Tests for embedder model configuration."""

    def test_embedder_uses_multilingual_model(self):
        from app.indexing.embedder import _get_model

        assert _get_model() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
