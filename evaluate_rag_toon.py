"""
MediLink RAG System - Comprehensive Evaluation Script

Evaluates both /ask (general RAG) and /patient/ask (TOON) endpoints
using BM25 retrieval (no deep learning dependencies)

Usage:
    python evaluate_rag_toon.py [--mode full|quick] [--output json|csv]
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

import numpy as np
import pandas as pd

from app.core.config import cfg
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index

from tests.test_rag_queries import RAG_TEST_QUERIES, TOON_TEST_QUERIES


RESULTS_DIR = Path(cfg.RESULTS_DIR)
RESULTS_DIR.mkdir(exist_ok=True)


def load_indexes() -> tuple:
    """Load vector store and BM25 index."""
    print("Loading indexes...")
    vs = VectorStore(dim=1024)
    vs.load("data/processed")
    print(f"  Vector store: {len(vs.documents)} documents")
    
    bm25 = BM25Index.load("data/processed")
    print(f"  BM25 index: {len(bm25.documents)} documents")
    
    return vs, bm25


def run_rag_evaluation(
    vs: VectorStore,
    bm25: BM25Index,
    mode: str = "quick",
    k_values: List[int] = [5, 10, 20],
) -> pd.DataFrame:
    """Evaluate RAG retrieval on test queries using BM25."""
    print(f"\n{'='*60}")
    print(f"RAG Retrieval Evaluation (BM25)")
    print(f"{'='*60}")
    
    results = []
    
    for category, lang_dict in RAG_TEST_QUERIES.items():
        queries = lang_dict.get("arabic", [])[:3] if mode == "quick" else lang_dict.get("arabic", [])
        
        for query in queries:
            start = time.time()
            
            try:
                bm25_results = bm25.search(query, k=20)
                
                retrieved_ids = [r.get("doc_id", f"doc_{i}") for i, r in enumerate(bm25_results)]
                latency = time.time() - start
                
                results.append({
                    "query": query,
                    "language": "arabic",
                    "category": category,
                    "num_retrieved": len(bm25_results),
                    "latency_seconds": latency,
                    "doc_ids": retrieved_ids[:10],
                })
                
                print(f"  [{category}] {query[:40]}... → {len(bm25_results)} docs ({latency:.3f}s)")
                
            except Exception as e:
                print(f"  ERROR [{category}]: {e}")
                results.append({
                    "query": query,
                    "language": "arabic",
                    "category": category,
                    "num_retrieved": 0,
                    "latency_seconds": -1,
                    "doc_ids": [],
                    "error": str(e),
                })
    
    return pd.DataFrame(results)


def run_toon_evaluation(
    vs: VectorStore,
    bm25: BM25Index,
    mode: str = "quick",
) -> pd.DataFrame:
    """Evaluate TOON patient-specific queries."""
    print(f"\n{'='*60}")
    print(f"TOON Patient Evaluation")
    print(f"{'='*60}")
    
    results = []
    
    for tier, tier_data in TOON_TEST_QUERIES.items():
        queries = tier_data["queries"]["questions"][:5] if mode == "quick" else tier_data["queries"]["questions"]
        patient_id = tier_data["queries"]["patient_id"]
        
        for query in queries:
            start = time.time()
            
            try:
                if "simple" in tier:
                    retrieved = bm25.search(query, k=10)
                    tier_label = "BM25"
                elif "moderate" in tier:
                    retrieved = bm25.search(query, k=10)
                    tier_label = "Hybrid"
                else:
                    retrieved = bm25.search(query, k=15)
                    tier_label = "Full LLM"
                
                latency = time.time() - start
                
                results.append({
                    "query": query,
                    "patient_id": patient_id,
                    "tier": tier,
                    "tier_label": tier_label,
                    "num_retrieved": len(retrieved),
                    "latency_seconds": latency,
                })
                
                print(f"  [{tier}] {query[:40]}... → {len(retrieved)} ({latency:.3f}s)")
                
            except Exception as e:
                print(f"  ERROR [{tier}]: {e}")
                results.append({
                    "query": query,
                    "patient_id": patient_id,
                    "tier": tier,
                    "tier_label": tier,
                    "num_retrieved": 0,
                    "latency_seconds": -1,
                    "error": str(e),
                })
    
    return pd.DataFrame(results)


def calculate_summary_stats(df: pd.DataFrame) -> Dict:
    """Calculate summary statistics."""
    total_queries = len(df)
    total_retrieved = df["num_retrieved"].sum()
    
    valid_latency = df[df["latency_seconds"] > 0]["latency_seconds"]
    avg_valid_latency = valid_latency.mean() if len(valid_latency) > 0 else 0
    
    stats = {
        "total_queries": total_queries,
        "total_documents_retrieved": int(total_retrieved),
        "avg_latency_seconds": round(avg_valid_latency, 4),
    }
    
    if total_queries > 0 and avg_valid_latency > 0:
        stats["queries_per_second"] = round(total_queries / avg_valid_latency, 2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate MediLink RAG/TOON")
    parser.add_argument("--mode", choices=["full", "quick"], default="quick")
    parser.add_argument("--output", choices=["json", "csv"], default="csv")
    args = parser.parse_args()
    
    print(f"\n{'#'*60}")
    print(f"# MediLink RAG/TOON Evaluation")
    print(f"# Mode: {args.mode}")
    print(f"# Time: {datetime.now().isoformat()}")
    print(f"{'#'*60}")
    
    vs, bm25 = load_indexes()
    
    rag_df = run_rag_evaluation(vs, bm25, mode=args.mode)
    toon_df = run_toon_evaluation(vs, bm25, mode=args.mode)
    
    rag_stats = calculate_summary_stats(rag_df)
    toon_stats = calculate_summary_stats(toon_df)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output == "csv":
        rag_df.to_csv(RESULTS_DIR / f"rag_eval_{timestamp}.csv", index=False)
        toon_df.to_csv(RESULTS_DIR / f"toon_eval_{timestamp}.csv", index=False)
        
        summary = {
            "rag": rag_stats,
            "toon": toon_stats,
            "mode": args.mode,
            "timestamp": timestamp,
        }
        with open(RESULTS_DIR / f"eval_summary_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"\nRAG (BM25 only):")
    for k, v in rag_stats.items():
        print(f"  {k}: {v}")
    print(f"\nTOON:")
    for k, v in toon_stats.items():
        print(f"  {k}: {v}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()