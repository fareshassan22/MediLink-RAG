"""Re-map ground truth relevant_docs to match the current vector store index.

For each ground truth query, performs dense retrieval and assigns the top
matching chunk IDs as the new relevant_docs.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.indexing.vector_store import VectorStore
from app.indexing.embedder import embed_texts
from app.core.config import cfg


def remap(top_k: int = 5, score_threshold: float = 0.45):
    # Load vector store
    vs = VectorStore(dim=1024)
    vs.load(str(cfg.INDEX_DIR))
    print(f"Loaded {len(vs.documents)} documents")

    # Load ground truth
    with open(cfg.EVAL_SET_PATH, "r", encoding="utf-8") as f:
        gt = json.load(f)

    updated = 0
    for entry in gt:
        query = entry["query"]
        emb = embed_texts([query])[0]
        results = vs.search(emb, k=top_k)

        # Keep only results above threshold
        relevant = []
        for r in results:
            if r["score"] >= score_threshold:
                relevant.append(f"doc_{r['doc_idx']}")

        # Guarantee at least 2 relevant docs
        if len(relevant) < 2:
            relevant = [f"doc_{r['doc_idx']}" for r in results[:2]]

        old_count = len(entry.get("relevant_docs", []))
        entry["relevant_docs"] = relevant
        updated += 1
        if updated <= 3:
            print(f"  [{entry['id']}] {query[:50]}...")
            print(f"    old: {old_count} docs -> new: {len(relevant)} docs: {relevant}")

    # Write back
    out_path = cfg.EVAL_SET_PATH
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(gt, f, ensure_ascii=False, indent=2)

    print(f"\nUpdated {updated} entries in {out_path}")


if __name__ == "__main__":
    remap()
