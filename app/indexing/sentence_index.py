from typing import List, Dict
import re
import pickle
import os


class SentenceIndex:
    """Split chunks into sentences for finer-grained retrieval"""

    def __init__(self, sentences: List[Dict] | None = None):
        self.sentences = sentences or []

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Simple sentence splitter supporting English and Arabic."""
        # Split on periods, exclamation, question marks (English + Arabic)
        sentences = re.split(r'(?<=[.!?؟\n])\s+', text)
        # Also split on newlines
        result = []
        for s in sentences:
            for part in s.split('\n'):
                part = part.strip()
                if len(part) > 10:
                    result.append(part)
        return result

    def build_from_chunks(self, chunks: List[Dict]) -> None:
        """Build sentence-level index from chunks"""
        self.sentences = []

        for chunk_idx, chunk in enumerate(chunks):
            text = chunk.get("text", "")

            # Simple sentence splitting on . ! ?
            sentences = self._split_sentences(text)

            for sent_idx, sentence in enumerate(sentences):
                if sentence.strip():
                    self.sentences.append(
                        {
                            "sentence": sentence,
                            "chunk_idx": chunk_idx,
                            "sent_idx": sent_idx,
                            "metadata": chunk.get("metadata", {}),
                            "page": chunk.get("page"),
                            "source": chunk.get("source"),
                        }
                    )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        with open(f"{path}/sentences.pkl", "wb") as f:
            pickle.dump(self.sentences, f)

    @staticmethod
    def load(path: str) -> "SentenceIndex":
        with open(f"{path}/sentences.pkl", "rb") as f:
            sentences = pickle.load(f)
        return SentenceIndex(sentences)

    def search(self, query: str, k: int = 10) -> List[Dict]:
        """Lightweight search: returns top-k sentences by simple substring match score.

        For production evaluation use the retrieval pipeline with embeddings + reranker.
        This helper is convenience only.
        """
        q = query.lower()
        scored = []
        for s in self.sentences:
            score = 1.0 if q in s["sentence"].lower() else 0.0
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]
