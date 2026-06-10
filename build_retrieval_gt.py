#!/usr/bin/env python3
"""Build an HONEST retrieval ground truth for the global medical-textbook RAG.

Why this script exists
----------------------
The old data/eval_ground_truth.json was *circular*: its "relevant_docs" were
simply the top-10 results of a previous hybrid-retrieval run (note its
`retrieval_method: hybrid_retrieval`, `total_retrieved: 10`, and exactly 10
relevant docs per query). Measuring retrieval against that only asks "can you
reproduce a prior run?" — it cannot tell us whether retrieval is actually good.
It also keyed relevance on positional ids (doc_2229) that change whenever the
index is rebuilt.

This builder fixes both problems:
  1. Relevance is decided by an INDEPENDENT LLM judge (Gemini), not by the
     retrieval pipeline under test.
  2. Candidates come from a simple cosine-similarity pre-filter (top-N), which
     is independent of the fusion / reranking we are evaluating. This is the
     standard TREC-style "pooling" approach.
  3. Every relevant chunk is kept (NO artificial top-10 cap), so the count of
     relevant docs varies per query and reflects the corpus.
  4. Docs are keyed by a STABLE content id: {source}_p{page}_c{chunk_id}
     (the same scheme the BM25 index uses), so the ground truth survives a
     re-index.

Output: data/eval_ground_truth_judged.json
  [{id, query, language, category, difficulty,
    relevant_docs: [stable_id, ...], n_candidates, n_relevant,
    annotation_method}]

Run (Colab, GPU for embeddings, Gemini for judging):
    GEMINI_API_KEY=... python3 build_retrieval_gt.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

# ── Config ────────────────────────────────────────────────────────────
DOCS_PATH = Path("data/processed/docs.jsonl")
QUERIES_PATH = Path("data/eval_ground_truth.json")       # reuse the QUERIES only
OUTPUT_PATH = Path("data/eval_ground_truth_judged.json")

EMBED_MODEL = os.getenv("GT_EMBED_MODEL", "BAAI/bge-m3")
JUDGE_MODEL = os.getenv("GT_JUDGE_MODEL", "gemini-2.0-flash")
PREFILTER_TOP_K = int(os.getenv("GT_PREFILTER_K", "50"))  # candidate pool depth
JUDGE_BATCH_SIZE = int(os.getenv("GT_JUDGE_BATCH", "20"))  # chunks per judge call
TEXT_PREVIEW_CHARS = 320


def _stable_id(meta: dict) -> str:
    """Content-stable id matching the BM25 index scheme."""
    return f"{meta.get('source')}_p{meta.get('page')}_c{meta.get('chunk_id')}"


# ── Load corpus + queries ─────────────────────────────────────────────
def load_docs() -> tuple[list[str], list[str]]:
    if not DOCS_PATH.exists():
        raise SystemExit(f"Missing {DOCS_PATH}. Build the index first (index_book.py).")
    texts, ids = [], []
    with open(DOCS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            texts.append(row["text"])
            ids.append(_stable_id(row.get("metadata", {})))
    if len(set(ids)) != len(ids):
        raise SystemExit("Stable ids are not unique — cannot build a reliable GT.")
    return texts, ids


def load_queries() -> list[dict]:
    with open(QUERIES_PATH, encoding="utf-8") as f:
        return json.load(f)


# ── Gemini judge ──────────────────────────────────────────────────────
def get_gemini():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set — required for the judge.")
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel(JUDGE_MODEL)


_JUDGE_SYSTEM = (
    "You are a precise medical relevance judge. Given a query and a list of "
    "text chunks from a medical encyclopedia, decide for EACH chunk whether it "
    "contains information that helps answer the query (symptoms, causes, "
    "treatment, diagnosis, definitions on that topic). Chunks about a different "
    "topic are NOT relevant. Respond with ONLY a JSON array of booleans, one "
    "per chunk, in order."
)


def judge_batch(model, query: str, chunk_texts: list[str]) -> list[bool]:
    """Return a relevance bool per chunk. Retries; conservative on failure."""
    listing = "".join(
        f"\n[{i + 1}] {t[:TEXT_PREVIEW_CHARS]}\n" for i, t in enumerate(chunk_texts)
    )
    prompt = (
        f"{_JUDGE_SYSTEM}\n\nQuery: {query}\n\nChunks:{listing}\n\n"
        f"Respond with a JSON array of exactly {len(chunk_texts)} booleans."
    )
    for attempt in range(4):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "max_output_tokens": 600,
                    "response_mime_type": "application/json",
                },
            )
            raw = (resp.text or "").strip()
            start, end = raw.find("["), raw.rfind("]") + 1
            arr = json.loads(raw[start:end])
            if isinstance(arr, list) and len(arr) == len(chunk_texts):
                return [bool(x) for x in arr]
        except Exception as e:
            msg = str(e).lower()
            if any(t in msg for t in ("rate", "quota", "429", "resource_exhausted")):
                wait = 2 ** attempt + 1
                print(f"    judge rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"    judge error: {type(e).__name__}: {str(e)[:80]}")
            time.sleep(1)
    # Conservative: if we genuinely could not judge, mark none relevant rather
    # than fabricating relevance. This UNDER-counts, never invents.
    print("    WARNING: judge failed for a batch — marking all not-relevant.")
    return [False] * len(chunk_texts)


# ── Main ──────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading corpus + queries...")
    doc_texts, doc_ids = load_docs()
    queries = load_queries()
    print(f"  {len(doc_texts)} docs, {len(queries)} queries")

    print(f"Embedding with {EMBED_MODEL} (cosine pre-filter, independent of fusion)...")
    from sentence_transformers import SentenceTransformer
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBED_MODEL, device=device)
    doc_emb = embedder.encode(
        doc_texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
    )
    q_emb = embedder.encode(
        [q["query"] for q in queries], batch_size=64, normalize_embeddings=True
    )

    sims = q_emb @ doc_emb.T  # (n_queries, n_docs)
    del embedder
    import gc

    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    model = get_gemini()
    print(f"Judging with {JUDGE_MODEL} (independent of the retrieval pipeline)...")

    out: list[dict] = []
    t0 = time.time()
    for qi, q in enumerate(queries):
        top_idx = np.argsort(sims[qi])[::-1][:PREFILTER_TOP_K]
        cand_ids = [doc_ids[i] for i in top_idx]
        cand_txt = [doc_texts[i] for i in top_idx]

        relevant: list[str] = []
        for b in range(0, len(cand_txt), JUDGE_BATCH_SIZE):
            verdicts = judge_batch(model, q["query"], cand_txt[b : b + JUDGE_BATCH_SIZE])
            for cid, ok in zip(cand_ids[b : b + JUDGE_BATCH_SIZE], verdicts):
                if ok:
                    relevant.append(cid)

        out.append(
            {
                "id": q.get("id", qi),
                "query": q["query"],
                "language": q.get("language", "unknown"),
                "category": q.get("category", "unknown"),
                "difficulty": q.get("difficulty", "unknown"),
                "relevant_docs": relevant,          # ALL judged-relevant, no cap
                "n_candidates": int(len(cand_ids)),
                "n_relevant": int(len(relevant)),
                "annotation_method": f"cosine_prefilter_top{PREFILTER_TOP_K}+{JUDGE_MODEL}_judge",
            }
        )
        rate = (qi + 1) / (time.time() - t0) * 60
        print(
            f"  [{qi + 1:3d}/{len(queries)}] {q['query'][:42]!r} "
            f"-> {len(relevant)}/{len(cand_ids)} relevant  ({rate:.1f} q/min)"
        )

        if (qi + 1) % 5 == 0:
            OUTPUT_PATH.write_text(
                json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    OUTPUT_PATH.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    counts = [e["n_relevant"] for e in out]
    n_empty = sum(1 for c in counts if c == 0)
    print(f"\nDone. Wrote {OUTPUT_PATH}")
    print(f"  relevant/query: min {min(counts)} max {max(counts)} "
          f"mean {sum(counts)/len(counts):.1f}")
    print(f"  queries with ZERO relevant (judge found none in pool): {n_empty}")
    if n_empty:
        print("  NOTE: zero-relevant queries are excluded from metrics by the "
              "evaluator (no gold = unmeasurable), which is the honest choice.")


if __name__ == "__main__":
    main()
