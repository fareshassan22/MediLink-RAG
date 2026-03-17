import re
from typing import List

import numpy as np

from app.indexing.embedder import embed_texts


def split_into_sentences(text: str) -> List[str]:
    sentence_endings = re.compile(r"(?<=[.!?])\s+")
    sentences = sentence_endings.split(text)
    return [s.strip() for s in sentences if s.strip()]


def split_into_paragraphs(text: str) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def semantic_chunk(
    text: str,
    chunk_size: int = 200,
    overlap: int = 40,
    min_paragraph_length: int = 40,
) -> List[str]:
    """Semantic chunking using paragraph-level splitting.

    Args:
        text: Input text to chunk.
        chunk_size: Target max words per chunk.
        overlap: Number of overlapping words between chunks.
        min_paragraph_length: Minimum characters for a paragraph to be kept.

    Returns:
        List of semantic chunks.
    """
    paragraphs = split_into_paragraphs(text)

    if len(paragraphs) <= 1:
        return _chunk_by_sentences(text, chunk_size, overlap)

    filtered_paragraphs = [p for p in paragraphs if len(p) >= min_paragraph_length]

    if not filtered_paragraphs:
        return _chunk_by_sentences(text, chunk_size, overlap)

    if len(filtered_paragraphs) == 1:
        return _chunk_by_sentences(filtered_paragraphs[0], chunk_size, overlap)

    chunks = []
    current_chunk = []
    current_length = 0

    for para in filtered_paragraphs:
        para_len = len(para.split())

        if current_length + para_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = _apply_overlap(
                current_chunk, current_length, overlap
            )

        current_chunk.append(para)
        current_length += para_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


def _apply_overlap(
    current_chunk: List[str], current_length: int, overlap: int
) -> tuple[List[str], int]:
    """Apply overlap to the current chunk."""
    if overlap <= 0:
        return [], 0

    overlap_words = " ".join(current_chunk).split()
    overlap_sentences = []
    overlap_len = 0

    for s in reversed(current_chunk):
        s_len = len(s.split())
        if overlap_len + s_len <= overlap:
            overlap_sentences.insert(0, s)
            overlap_len += s_len
        else:
            break

    return overlap_sentences, overlap_len


def _chunk_by_sentences(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fallback: chunk by sentences with word limit."""
    sentences = split_into_sentences(text)

    if not sentences:
        return [text] if text else []

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence.split())

        if current_length + sentence_len > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk, current_length = _apply_overlap(
                current_chunk, current_length, overlap
            )

        current_chunk.append(sentence)
        current_length += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]
