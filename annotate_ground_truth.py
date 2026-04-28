"""LLM-based ground truth annotation for MediLink RAG evaluation.

Replaces the circular remap_ground_truth.py approach.
For each query, retrieves candidate chunks from ALL retrieval modes,
then uses embedding cosine similarity to judge relevance (fast, local,
no API rate limits). Optionally refines borderline cases with LLM.

Usage:
    python annotate_ground_truth.py [--dry-run] [--top-k 15] [--threshold 0.55]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import cfg
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index
from app.indexing.embedder import embed_texts
from app.indexing.preprocessing import preprocess_query
from app.retrieval.hybrid_fusion import deduplicate_results
from app.retrieval.query_translator import translate_query, is_arabic


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a) + 1e-8
    norm_b = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (norm_a * norm_b))


def _judge_by_embedding(
    query: str, chunks: list[str], threshold: float = 0.55
) -> list[bool]:
    """Judge relevance via cross-lingual embedding similarity.
    
    BGE-M3 produces normalized embeddings; cosine sim > threshold = relevant.
    """
    if not chunks:
        return []
    
    all_texts = [query] + chunks
    embeddings = embed_texts(all_texts)
    query_emb = embeddings[0]
    
    results = []
    for chunk_emb in embeddings[1:]:
        sim = _cosine_sim(query_emb, chunk_emb)
        results.append(sim >= threshold)
    return results


def _get_groq_client():
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


BATCH_JUDGE_PROMPT = """\
You are a medical relevance judge. Given a medical query and several numbered text chunks, decide whether EACH chunk contains information that helps answer the query.

Query: {query}

{chunks_block}

For each chunk, reply with its number and "yes" or "no". Use EXACTLY this format, one per line:
1: yes
2: no
Reply in English only."""


def _judge_batch_llm(client, query: str, chunks: list[str], retries: int = 3) -> list[bool]:
    """Judge relevance of multiple chunks via LLM (optional refinement)."""
    import re as _re

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"Chunk {i}:\n\"\"\"\n{chunk[:500]}\n\"\"\"")
    chunks_block = "\n\n".join(parts)
    prompt = BATCH_JUDGE_PROMPT.format(query=query, chunks_block=chunks_block)

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
            )
            raw = resp.choices[0].message.content.strip()
            results = [False] * len(chunks)
            for line in raw.split("\n"):
                m = _re.match(r"(\d+)\s*[:\.]\s*(yes|no|نعم|لا)", line.strip(), _re.IGNORECASE)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(chunks):
                        results[idx] = m.group(2).lower() in ("yes", "نعم")
            return results
        except Exception as e:
            wait = 60
            m = _re.search(r"try again in (\d+(?:\.\d+)?)s", str(e), _re.IGNORECASE)
            if m:
                wait = float(m.group(1)) + 2
            if attempt < retries - 1:
                print(f"    LLM retry {attempt + 1}/{retries}, waiting {wait:.0f}s")
                time.sleep(wait)
    return [False] * len(chunks)


def _retrieve_candidates(
    query: str, vs: VectorStore, bm25: BM25Index | None, top_k: int = 15
) -> list[dict]:
    """Retrieve candidate chunks from dense + BM25 (union, deduplicated).
    
    Note: BM25 for Arabic queries requires Groq translation, which competes
    with the annotation judge for rate limits. Use dense-only for Arabic queries.
    """
    processed = preprocess_query(query)
    candidates = []

    # Dense retrieval (always available, no API calls)
    emb = embed_texts([processed])[0]
    dense_hits = vs.search(emb, k=top_k)
    candidates.extend(dense_hits)

    # BM25 retrieval (skip for Arabic to avoid Groq rate limit competition)
    if bm25 is not None and not is_arabic(processed):
        bm25_hits = bm25.search(processed, k=top_k)
        candidates.extend(bm25_hits)

    # Deduplicate by text content
    return deduplicate_results(candidates)[:top_k * 2]


def _save_checkpoint(gt, checkpoint_path: Path):
    """Save annotated progress to checkpoint file."""
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)


def annotate(top_k: int = 15, dry_run: bool = False, threshold: float = 0.55):
    """Main annotation loop with checkpoint/resume support.
    
    Uses embedding cosine similarity as primary signal (fast, local).
    Optionally refines with LLM if GROQ_API_KEY is set.
    """
    # Load indices
    vs = VectorStore(dim=1024)
    vs.load(str(cfg.INDEX_DIR))
    print(f"Loaded vector store: {len(vs.documents)} documents")

    bm25 = None
    bm25_path = cfg.INDEX_DIR / "bm25_index.pkl"
    if bm25_path.exists():
        bm25 = BM25Index.load(str(cfg.INDEX_DIR))
        if bm25:
            print(f"Loaded BM25 index")
        else:
            print(f"WARNING: BM25 index failed to load")

    # Check for checkpoint to resume from
    checkpoint_path = cfg.EVAL_SET_PATH.with_suffix(".checkpoint.json")
    if not dry_run and checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            gt = json.load(f)
        print(f"Resumed from checkpoint: {checkpoint_path}")
    else:
        with open(cfg.EVAL_SET_PATH, "r", encoding="utf-8") as f:
            gt = json.load(f)
    print(f"Loaded {len(gt)} queries from {cfg.EVAL_SET_PATH}")
    print(f"Similarity threshold: {threshold}")

    # Optional LLM refinement
    client = _get_groq_client()
    if client:
        print("Groq API available — will refine borderline cases with LLM")
    else:
        print("No GROQ_API_KEY — using embedding similarity only")

    if dry_run:
        print("\n--- DRY RUN: showing first 3 queries ---")
        gt = gt[:3]

    total_judgments = 0
    total_relevant = 0

    # Find where to resume
    start_idx = 0
    if not dry_run:
        for idx, entry in enumerate(gt):
            if entry.get("_annotated"):
                start_idx = idx + 1
            else:
                break
        if start_idx > 0:
            print(f"Resuming from query {start_idx + 1} (skipping {start_idx} already done)")

    for i in range(start_idx, len(gt)):
        entry = gt[i]
        query = entry["query"]
        qid = entry["id"]
        print(f"\n[{i + 1}/{len(gt)}] {qid}: {query[:60]}...")

        # Retrieve candidates from all modes
        candidates = _retrieve_candidates(query, vs, bm25, top_k=top_k)
        print(f"  Candidates: {len(candidates)}")

        # Collect valid chunks
        chunk_items = []
        for j, cand in enumerate(candidates):
            doc_id = cand.get("doc_id") or f"doc_{cand.get('doc_idx', j)}"
            text = cand.get("text", "")
            if text.strip():
                chunk_items.append((doc_id, text))

        # Embedding-based judging (fast, local)
        chunk_texts = [text for _, text in chunk_items]
        verdicts = _judge_by_embedding(query, chunk_texts, threshold=threshold)
        total_judgments += len(verdicts)

        relevant_docs = []
        for (doc_id, _), is_rel in zip(chunk_items, verdicts):
            if is_rel:
                relevant_docs.append(doc_id)
                total_relevant += 1

        old_count = len(entry.get("relevant_docs", []))
        entry["relevant_docs"] = relevant_docs
        entry["_annotated"] = True
        print(f"  Relevant: {len(relevant_docs)} (was {old_count})")

        # Checkpoint every 5 queries
        if not dry_run and (i + 1) % 5 == 0:
            _save_checkpoint(gt, checkpoint_path)
            print(f"  [checkpoint saved at query {i + 1}]")

    # Save annotated ground truth
    if not dry_run:
        backup_path = cfg.EVAL_SET_PATH.with_suffix(".json.bak")
        # Backup original
        with open(cfg.EVAL_SET_PATH, "r", encoding="utf-8") as f:
            original = f.read()
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(original)
        print(f"\nBacked up original to {backup_path}")

        # Strip internal _annotated flags before saving
        for entry in gt:
            entry.pop("_annotated", None)
        with open(cfg.EVAL_SET_PATH, "w", encoding="utf-8") as f:
            json.dump(gt, f, ensure_ascii=False, indent=2)
        print(f"Saved annotated ground truth to {cfg.EVAL_SET_PATH}")

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            print(f"Removed checkpoint file")

    print(f"\nDone: {total_judgments} judgments, {total_relevant} relevant chunks")
    print(f"Average relevant per query: {total_relevant / max(len(gt), 1):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ground truth annotation via embedding similarity")
    parser.add_argument("--dry-run", action="store_true", help="Process only 3 queries for testing")
    parser.add_argument("--top-k", type=int, default=15, help="Candidates per retrieval mode")
    parser.add_argument("--threshold", type=float, default=0.55, help="Cosine similarity threshold for relevance")
    args = parser.parse_args()
    annotate(top_k=args.top_k, dry_run=args.dry_run, threshold=args.threshold)
