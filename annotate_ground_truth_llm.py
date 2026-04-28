#!/usr/bin/env python3
"""
Annotate ground truth using a two-stage approach on GPU 3:
  Stage 1: BGE-M3 embedding similarity → top-50 candidates per query
  Stage 2: Qwen2.5-32B-Instruct (4-bit) judges those 50 candidates

This is NOT circular because:
- Stage 1 uses raw cosine similarity (no fusion/filtering/reranking)
- Stage 2 uses an independent LLM judge, not the retrieval pipeline
"""

import json
import os
import time
import gc
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# ── Config ────────────────────────────────────────────────────────────
MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"
GT_PATH = "data/eval_ground_truth.json"
DOCS_PATH = "data/processed/docs.jsonl"
OUTPUT_PATH = "data/eval_ground_truth.json"

PREFILTER_TOP_K = 50      # embedding pre-filter candidates
LLM_BATCH_SIZE = 25       # chunks per LLM call
FINAL_TOP_K = 10           # keep top-10 relevant docs per query


# ── Load data ─────────────────────────────────────────────────────────
print("Loading documents...")
docs = []
with open(DOCS_PATH) as f:
    for line in f:
        line = line.strip()
        if line:
            docs.append(json.loads(line))
doc_texts = [d["text"] for d in docs]
doc_ids = [d["doc_id"] for d in docs]
print(f"  {len(docs)} documents loaded")

print("Loading queries...")
with open(GT_PATH) as f:
    queries = json.load(f)
query_texts = [q["query"] for q in queries]
print(f"  {len(queries)} queries loaded")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: Embedding pre-filter
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STAGE 1: Embedding pre-filter (BGE-M3)")
print(f"{'='*60}")

from sentence_transformers import SentenceTransformer

print(f"Loading {EMBED_MODEL}...")
embedder = SentenceTransformer(EMBED_MODEL, device="cuda:0")

print("Embedding documents...")
t0 = time.time()
doc_embeddings = embedder.encode(
    doc_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"  Doc embeddings: {doc_embeddings.shape} in {time.time()-t0:.1f}s")

print("Embedding queries...")
t0 = time.time()
query_embeddings = embedder.encode(
    query_texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,
)
print(f"  Query embeddings: {query_embeddings.shape} in {time.time()-t0:.1f}s")

# Compute similarity matrix and get top-K candidates per query
print(f"Computing similarity -> top-{PREFILTER_TOP_K} candidates per query...")
sim_matrix = query_embeddings @ doc_embeddings.T  # (99, 759)

candidates_per_query = {}
for qi in range(len(queries)):
    scores = sim_matrix[qi]
    top_indices = np.argsort(scores)[::-1][:PREFILTER_TOP_K]
    candidates_per_query[qi] = [(int(idx), float(scores[idx])) for idx in top_indices]
    if qi < 3:
        print(f"  Query {qi}: top scores = {[f'{s:.3f}' for _, s in candidates_per_query[qi][:5]]}")

# Free embedding model VRAM
del embedder, doc_embeddings, query_embeddings, sim_matrix
gc.collect()
torch.cuda.empty_cache()
print("  Embedder freed from GPU")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: LLM Judge
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("STAGE 2: LLM Judge (Qwen2.5-32B-Instruct, 4-bit)")
print(f"{'='*60}")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(f"Loading {MODEL_NAME} (4-bit)...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map={"": 0},
    trust_remote_code=True,
)
model.eval()
print(f"  Model loaded in {time.time()-t0:.1f}s")


def judge_batch(query: str, chunk_batch: list) -> list:
    """Ask LLM to judge relevance of each chunk to the query."""
    chunks_text = ""
    for i, (doc_idx, sim_score) in enumerate(chunk_batch):
        text_preview = docs[doc_idx]["text"][:300]
        chunks_text += f"\n[{i+1}] {text_preview}\n"

    prompt = f"""You are a medical relevance judge. Given a medical query and text chunks from a medical encyclopedia, determine which chunks are relevant to answering the query.

RELEVANT = contains information that helps answer the query (symptoms, treatment, causes, diagnosis, definitions related to the topic).
NOT RELEVANT = discusses unrelated topics.

Query: {query}

Chunks:{chunks_text}

Respond with ONLY a JSON array of {len(chunk_batch)} booleans (true/false), one per chunk.
Example: [true, false, true, false]

Answer:"""

    messages = [
        {"role": "system", "content": "You are a precise medical relevance judge. Respond only with a JSON boolean array."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    try:
        start = response.index("[")
        end = response.index("]") + 1
        results = json.loads(response[start:end])
        if len(results) == len(chunk_batch):
            return [bool(r) for r in results]
    except (ValueError, json.JSONDecodeError):
        pass

    print(f"    WARNING Parse fail: {response[:80]}")
    return [False] * len(chunk_batch)


# ── Annotation loop ──────────────────────────────────────────────────
batches_per_query = (PREFILTER_TOP_K + LLM_BATCH_SIZE - 1) // LLM_BATCH_SIZE
total_calls = len(queries) * batches_per_query
print(f"\nAnnotating: {len(queries)} queries x {batches_per_query} batches = ~{total_calls} LLM calls")
print()

annotated = []
total_start = time.time()

for qi, q in enumerate(queries):
    query_text = q["query"]
    candidates = candidates_per_query[qi]
    query_start = time.time()
    relevant_doc_ids = []

    # Process candidates in batches
    for batch_start in range(0, len(candidates), LLM_BATCH_SIZE):
        batch = candidates[batch_start:batch_start + LLM_BATCH_SIZE]
        results = judge_batch(query_text, batch)

        for (doc_idx, sim_score), is_relevant in zip(batch, results):
            if is_relevant:
                relevant_doc_ids.append(doc_ids[doc_idx])

    query_time = time.time() - query_start
    elapsed = time.time() - total_start
    rate = (qi + 1) / elapsed * 60

    entry = {
        "id": q["id"],
        "query": q["query"],
        "language": q["language"],
        "category": q["category"],
        "difficulty": q["difficulty"],
        "relevant_docs": relevant_doc_ids[:FINAL_TOP_K],
        "total_relevant_found": len(relevant_doc_ids),
        "annotation_method": "embedding_prefilter+qwen2.5-32b-judge",
    }
    annotated.append(entry)

    print(
        f"  [{qi+1:3d}/{len(queries)}] {q['id']} | "
        f"found {len(relevant_doc_ids):3d} relevant / {PREFILTER_TOP_K} candidates | "
        f"kept top-{min(FINAL_TOP_K, len(relevant_doc_ids))} | "
        f"{query_time:.1f}s | "
        f"{rate:.2f} q/min"
    )

    # Save progress every 5 queries
    if (qi + 1) % 5 == 0:
        with open(OUTPUT_PATH, "w") as f:
            json.dump(annotated, f, ensure_ascii=False, indent=2)
        print(f"    Progress saved ({qi+1}/{len(queries)})")

# ── Final save ────────────────────────────────────────────────────────
with open(OUTPUT_PATH, "w") as f:
    json.dump(annotated, f, ensure_ascii=False, indent=2)

total_time = time.time() - total_start
print(f"\nDone! Annotation complete!")
print(f"   Queries: {len(annotated)}")
print(f"   Time: {total_time/60:.1f} min")
print(f"   Output: {OUTPUT_PATH}")

relevance_counts = [e["total_relevant_found"] for e in annotated]
print(f"   Avg relevant docs per query: {sum(relevance_counts)/len(relevance_counts):.1f}")
print(f"   Min: {min(relevance_counts)}, Max: {max(relevance_counts)}")
