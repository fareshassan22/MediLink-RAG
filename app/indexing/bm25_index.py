from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from rank_bm25 import BM25Okapi

from app.indexing.preprocessing import preprocess_query, preprocess_document
from app.indexing.arabic_tokenizer import (
    tokenize_bilingual,
    tokenize_arabic,
    tokenize_english,
    is_arabic,
)

logger = logging.getLogger(__name__)


class BM25Index:
    """Lightweight BM25 wrapper with proper Arabic/English tokenization."""

    def __init__(self, use_preprocessing: bool = True) -> None:
        self.bm25: BM25Okapi | None = None
        self.corpus_tokens: List[List[str]] = []
        self.corpus: List[str] = []
        self.documents: Dict[str, Dict] = {}
        self.doc_id_list: List[str] = []  # Maps corpus index -> doc_id
        self.use_preprocessing = use_preprocessing

    @staticmethod
    def tokenize(text: str, preprocess: bool = True) -> List[str]:
        """Tokenize text with proper Arabic/English tokenization."""
        if preprocess:
            text = preprocess_document(text)

        if is_arabic(text):
            return tokenize_arabic(text, remove_stopwords=True)
        else:
            return tokenize_english(text, remove_stopwords=True)

    @staticmethod
    def tokenize_query(query: str) -> List[str]:
        """Tokenize query with bilingual support."""
        if is_arabic(query):
            return tokenize_arabic(query, remove_stopwords=True)
        else:
            return tokenize_english(query, remove_stopwords=True)

    def build(self, docs: List[str], force_preprocessing: bool = True) -> None:
        """Build index from documents."""
        self.corpus = docs
        self.corpus_tokens = [
            self.tokenize(d, preprocess=force_preprocessing or self.use_preprocessing)
            for d in docs
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        logger.info(f"Built BM25 index with {len(docs)} documents")

    def add_document(self, doc_id: str, text: str, metadata: Dict) -> None:
        """Add a single document to the index."""
        self.documents[doc_id] = {"text": text, "metadata": metadata}
        self.doc_id_list.append(doc_id)
        if self.bm25 is None:
            self.corpus = [text]
            self.corpus_tokens = [self.tokenize(text)]
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.corpus.append(text)
            self.corpus_tokens.append(self.tokenize(text))
            self.bm25 = BM25Okapi(self.corpus_tokens)

    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for a query."""
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built")
        preprocessed = preprocess_query(query) if self.use_preprocessing else query
        q_tokens = self.tokenize_query(preprocessed)
        logger.info(f"BM25 query tokens: {q_tokens}")
        return self.bm25.get_scores(q_tokens).tolist()

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Search the index and return top-k results with proper scoring."""
        scores = self.get_scores(query)
        idx_scores = list(enumerate(scores))
        idx_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in idx_scores[:k]:
            if idx < len(self.corpus):
                # Look up metadata using the doc_id mapping
                metadata = {}
                if idx < len(self.doc_id_list):
                    doc_id = self.doc_id_list[idx]
                    doc_entry = self.documents.get(doc_id, {})
                    metadata = doc_entry.get("metadata", {})
                results.append(
                    {
                        "text": self.corpus[idx],
                        "doc_idx": idx,
                        "score": score,
                        "bm25_score": score,
                        "metadata": metadata,
                        "page": metadata.get("page"),
                        "source": metadata.get("source", "Medical Textbook"),
                    }
                )
        return results

    def save(self, path: str | Path) -> None:
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        data = {
            "corpus": self.corpus,
            "corpus_tokens": self.corpus_tokens,
            "documents": self.documents,
            "doc_id_list": self.doc_id_list,
            "use_preprocessing": self.use_preprocessing,
        }
        with open(path / "bm25_index.pkl", "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: str | Path) -> Optional["BM25Index"]:
        """Load index from disk."""
        path = Path(path)
        try:
            with open(path / "bm25_index.pkl", "rb") as f:
                data = pickle.load(f)
            index = BM25Index(use_preprocessing=data.get("use_preprocessing", True))
            index.corpus = data["corpus"]
            index.corpus_tokens = data["corpus_tokens"]
            index.documents = data["documents"]
            # Rebuild doc_id_list from documents dict if not saved
            index.doc_id_list = data.get("doc_id_list", list(data["documents"].keys()))
            if index.corpus_tokens:
                index.bm25 = BM25Okapi(index.corpus_tokens)
            logger.info(f"Loaded BM25 index with {len(index.corpus)} documents")
            return index
        except FileNotFoundError:
            logger.warning(f"BM25 index not found at {path}")
            return None
