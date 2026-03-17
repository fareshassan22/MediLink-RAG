import pytest
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestEmbedder:
    """Tests for text embedding module."""

    @patch("app.indexing.embedder.SentenceTransformer")
    def test_embed_texts_returns_list(self, mock_model):
        from app.indexing.embedder import embed_texts

        mock_instance = Mock()
        mock_instance.encode.return_value = np.random.rand(2, 1024).astype("float32")
        mock_model.return_value = mock_instance

        texts = ["text1", "text2"]
        result = embed_texts(texts)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0].shape == (1024,)

    @patch("app.indexing.embedder.SentenceTransformer")
    def test_embed_texts_single(self, mock_model):
        from app.indexing.embedder import embed_texts

        mock_instance = Mock()
        mock_instance.encode.return_value = np.random.rand(1, 1024).astype("float32")
        mock_model.return_value = mock_instance

        texts = ["single text"]
        result = embed_texts(texts)

        assert len(result) == 1

    @patch("app.indexing.embedder.SentenceTransformer")
    def test_embed_texts_empty(self, mock_model):
        from app.indexing.embedder import embed_texts

        result = embed_texts([])

        assert result == []

    @patch("app.indexing.embedder.SentenceTransformer")
    def test_embed_texts_normalized(self, mock_model):
        from app.indexing.embedder import embed_texts

        mock_instance = Mock()
        mock_instance.encode.return_value = np.random.rand(1, 1024).astype("float32")
        mock_model.return_value = mock_instance

        embed_texts(["test"])

        mock_instance.encode.assert_called_once()
        call_kwargs = mock_instance.encode.call_args[1]
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

        docs = [
            {"text": "doc1", "metadata": {"page": 1}},
            {"text": "doc2", "metadata": {"page": 2}},
        ]

        vs.add_documents(docs)

        assert len(vs.documents) == 2


class TestBM25Index:
    """Tests for BM25 index."""

    def test_bm25_index_init(self):
        from app.indexing.bm25_index import BM25Index

        bm25 = BM25Index()

        assert bm25 is not None

    @patch("app.indexing.bm25_index.BM25Okapi")
    def test_bm25_index_search(self, mock_bm25):
        from app.indexing.bm25_index import BM25Index

        mock_instance = Mock()
        mock_instance.get_top_n.return_value = [
            ("doc1", 1.5),
            ("doc2", 1.2),
        ]

        bm25 = BM25Index()
        bm25.model = mock_instance

        results = bm25.search("test query", k=5)

        assert isinstance(results, list)


class TestSentenceIndex:
    """Tests for sentence-level indexing."""

    def test_sentence_index_init(self):
        from app.indexing.sentence_index import SentenceIndex

        sidx = SentenceIndex()

        assert hasattr(sidx, "sentences")

    def test_sentence_index_build_from_chunks(self):
        from app.indexing.sentence_index import SentenceIndex

        sidx = SentenceIndex()

        chunks = [
            {"text": "This is sentence one. This is sentence two."},
            {"text": "Another chunk with sentences."},
        ]

        sidx.build_from_chunks(chunks)

        assert len(sidx.sentences) > 0


class TestChunker:
    """Tests for text chunking."""

    def test_chunk_text_by_sentence(self):
        from app.indexing.chunker import chunk_text_by_sentence

        text = "This is sentence one. This is sentence two. This is sentence three."

        chunks = chunk_text_by_sentence(text, chunk_size=2)

        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_chunk_text_by_sentence_small(self):
        from app.indexing.chunker import chunk_text_by_sentence

        text = "Short text."

        chunks = chunk_text_by_sentence(text, chunk_size=5)

        assert isinstance(chunks, list)


class TestPreprocessing:
    """Tests for text preprocessing."""

    def test_clean_text(self):
        from app.indexing.preprocessing import clean_text

        text = "  Test   text  with   spaces  "
        cleaned = clean_text(text)

        assert "  " not in cleaned

    def test_normalize_arabic(self):
        from app.indexing.preprocessing import normalize_arabic

        text = "أَلْف"
        normalized = normalize_arabic(text)

        assert normalized is not None

    def test_remove_special_characters(self):
        from app.indexing.preprocessing import remove_special_chars

        text = "Test @#$% text!"
        cleaned = remove_special_chars(text)

        assert "@" not in cleaned
        assert "#" not in cleaned


class TestArabicTokenizer:
    """Tests for Arabic tokenization."""

    def test_arabic_tokenizer_init(self):
        from app.indexing.arabic_tokenizer import ArabicTokenizer

        tokenizer = ArabicTokenizer()

        assert tokenizer is not None

    def test_tokenize_arabic(self):
        from app.indexing.arabic_tokenizer import ArabicTokenizer

        tokenizer = ArabicTokenizer()

        text = "هذا نص عربي للاختبار"
        tokens = tokenizer.tokenize(text)

        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_remove_arabic_stopwords(self):
        from app.indexing.arabic_tokenizer import ArabicTokenizer

        tokenizer = ArabicTokenizer()

        text = "هذا هو النص"
        tokens = tokenizer.remove_stopwords(text.split())

        assert isinstance(tokens, list)


class TestIndexPipeline:
    """Tests for indexing pipeline."""

    @patch("app.indexing.pdf_loader.load_pdf")
    @patch("app.indexing.preprocessing.clean_text")
    @patch("app.indexing.chunker.chunk_text_by_sentence")
    @patch("app.indexing.embedder.embed_texts")
    def test_index_pipeline_runs(self, mock_embed, mock_chunk, mock_clean, mock_pdf):
        from app.indexing.index_pipeline import IndexPipeline

        mock_pdf.return_value = [("page 1", "text 1")]
        mock_clean.return_value = "cleaned text"
        mock_chunk.return_value = ["chunk1", "chunk2"]
        mock_embed.return_value = [np.random.rand(1024).astype("float32")]

        pipeline = IndexPipeline()

        assert pipeline is not None


class TestEmbedderModel:
    """Tests for embedder model configuration."""

    def test_embedder_uses_multilingual_model(self):
        from app.indexing.embedder import _model

        assert _model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
