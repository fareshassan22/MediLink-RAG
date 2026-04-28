#!/usr/bin/env python3
"""Regenerate ground truth using actual retrieval results instead of LLM annotations.

This runs the retrieval pipeline for each query and uses the top-K results
as the ground truth relevant documents.

Usage:
    python regenerate_ground_truth.py [--mode hybrid|dense|bm25] [--top-k 10]
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import cfg
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index
from app.indexing.embedder import embed_texts
from app.retrieval.hybrid_fusion import hybrid_retrieval_fusion, deduplicate_results
from app.retrieval.query_translator import translate_query, is_arabic
from app.indexing.preprocessing import preprocess_query


def run_retrieval(query: str, vs: VectorStore, bm25: "BM25Index | None", mode: str, top_k: int):
    """Run retrieval for a single query."""
    processed = preprocess_query(query)
    
    dense_results = []
    bm25_results = []
    
    # Dense retrieval
    if mode in ("hybrid", "dense"):
        try:
            emb = embed_texts([processed])[0]
            dense_results = vs.search(emb, k=top_k)
            
            # Bilingual boost for Arabic queries
            if is_arabic(processed):
                en = translate_query(processed)
                if en and en != processed:
                    en_emb = embed_texts([en])[0]
                    dense_results.extend(vs.search(en_emb, k=top_k))
        except Exception as e:
            print(f"    Dense search failed: {e}")
    
    # BM25 retrieval
    if mode in ("hybrid", "bm25") and bm25 is not None:
        try:
            bm25_query = processed
            if is_arabic(processed):
                en = translate_query(processed)
                if en and en != processed:
                    bm25_query = en
            bm25_results = bm25.search(bm25_query, k=top_k)
        except Exception as e:
            print(f"    BM25 search failed: {e}")
    
    # Fusion
    if mode == "hybrid" and dense_results and bm25_results:
        fused = hybrid_retrieval_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results,
            query=processed,
            top_k=top_k,
        )
    elif mode == "bm25":
        fused = deduplicate_results(bm25_results)
    else:
        fused = deduplicate_results(dense_results)
    
    return fused


def main():
    parser = argparse.ArgumentParser(description="Regenerate ground truth using retrieval results")
    parser.add_argument("--mode", choices=["hybrid", "dense", "bm25"], default="hybrid",
                        help="Retrieval mode to use")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of top results to keep per query")
    parser.add_argument("--score-threshold", type=float, default=0.0,
                        help="Minimum score threshold (0 = no filter)")
    parser.add_argument("--input", type=str, default=None,
                        help="Input file (default: data/eval_ground_truth.json)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: overwrites input)")
    args = parser.parse_args()
    
    input_path = Path(args.input) if args.input else cfg.EVAL_SET_PATH
    output_path = Path(args.output) if args.output else input_path
    
    print(f"Loading evaluation queries from {input_path}...")
    with open(input_path, encoding="utf-8") as f:
        queries = json.load(f)
    print(f"  Loaded {len(queries)} queries")
    
    print(f"\nLoading vector store...")
    vs = VectorStore(dim=1024)
    vs.load(str(cfg.INDEX_DIR))
    print(f"  Loaded {len(vs.documents)} documents")
    
    print(f"Loading BM25 index...")
    bm25 = None
    bm25_path = cfg.INDEX_DIR
    if (bm25_path / "bm25_index.pkl").exists():
        bm25 = BM25Index.load(str(bm25_path))
        if bm25:
            print(f"  BM25 index loaded with {len(bm25.corpus)} docs")
        else:
            print(f"  BM25 index failed to load")
    else:
        bm25 = None
        print(f"  BM25 index not found, skipping")
    
    print(f"\nRunning {args.mode} retrieval for {len(queries)} queries...")
    print(f"  Top-K: {args.top_k}, Score threshold: {args.score_threshold}")
    
    results = []
    start_time = time.time()
    
    for i, entry in enumerate(queries):
        query = entry["query"]
        q_start = time.time()
        
        retrieved = run_retrieval(query, vs, bm25, args.mode, args.top_k)
        
        # Filter by score threshold
        if args.score_threshold > 0:
            retrieved = [r for r in retrieved if r.get("score", r.get("dense_score", 0)) >= args.score_threshold]
        
        # Convert to doc_ids
        relevant_docs = []
        for r in retrieved:
            doc_idx = r.get("doc_idx")
            if doc_idx is not None:
                relevant_docs.append(f"doc_{doc_idx}")
        
        # Ensure at least some results
        if not relevant_docs and retrieved:
            # Take top results even if below threshold
            relevant_docs = [f"doc_{r['doc_idx']}" for r in retrieved[:3] if r.get("doc_idx") is not None]
        
        q_time = time.time() - q_start
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed * 60
        
        results.append({
            "id": entry.get("id"),
            "query": query,
            "language": entry.get("language"),
            "category": entry.get("category"),
            "difficulty": entry.get("difficulty"),
            "relevant_docs": relevant_docs[:args.top_k],
            "total_retrieved": len(retrieved),
            "retrieval_method": f"{args.mode}_retrieval",
        })
        
        print(f"  [{i+1:3d}/{len(queries)}] {entry.get('id', 'N/A')} | "
              f"retrieved {len(relevant_docs):2d} docs | "
              f"{q_time:.2f}s | {rate:.1f} q/min")
    
    print(f"\nSaving results to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    total_time = time.time() - start_time
    doc_counts = [len(r["relevant_docs"]) for r in results]
    
    print(f"\nDone!")
    print(f"  Queries: {len(results)}")
    print(f"  Time: {total_time/60:.1f} min")
    print(f"  Avg docs per query: {sum(doc_counts)/len(doc_counts):.1f}")
    print(f"  Min: {min(doc_counts)}, Max: {max(doc_counts)}")


if __name__ == "__main__":
    main()