#!/usr/bin/env python3
"""
Annotate TOON ground truth using BGE-M3 + LLM judge on GPUs 4,6
"""

import json
import os
import time
import gc
import torch
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from tests.test_rag_queries import TOON_TEST_QUERIES
from app.indexing.vector_store import VectorStore

MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct"
EMBED_MODEL = "BAAI/bge-m3"
OUTPUT_PATH = "data/toon_ground_truth.json"

PREFILTER_TOP_K = 30
LLM_BATCH_SIZE = 15
FINAL_TOP_K = 10

print("Loading documents...")
vs = VectorStore(dim=1024)
vs.load("data/processed")
docs = vs.documents
doc_texts = [d.text[:500] for d in docs]
doc_ids = [d.doc_id for d in docs]
print(f"  {len(docs)} documents loaded")

queries = []
for tier, tier_data in TOON_TEST_QUERIES.items():
    for q in tier_data["queries"]["questions"]:
        queries.append({
            "query": q,
            "tier": tier,
            "patient_id": tier_data["queries"]["patient_id"]
        })
print(f"  {len(queries)} TOON queries")

print(f"\n{'='*60}")
print("STAGE 1: Embedding pre-filter (BGE-M3)")
print(f"{'='*60}")

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer(EMBED_MODEL, device="cuda")

print("Embedding documents...")
t0 = time.time()
doc_embeddings = embedder.encode(doc_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
print(f"  Done in {time.time()-t0:.1f}s")

print("Embedding queries...")
query_texts = [q["query"] for q in queries]
query_embeddings = embedder.encode(query_texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

sim_matrix = query_embeddings @ doc_embeddings.T

candidates_per_query = {}
for qi in range(len(queries)):
    scores = sim_matrix[qi]
    top_indices = np.argsort(scores)[::-1][:PREFILTER_TOP_K]
    candidates_per_query[qi] = [(int(idx), float(scores[idx])) for idx in top_indices]

del embedder, doc_embeddings, query_embeddings, sim_matrix
gc.collect()
torch.cuda.empty_cache()

print(f"\n{'='*60}")
print("STAGE 2: LLM Judge")
print(f"{'='*60}")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print(f"Loading {MODEL_NAME} (4-bit)...")
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
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("  Model loaded")

def judge_batch(query: str, chunk_batch: list) -> list:
    chunks_text = ""
    for i, (doc_idx, sim_score) in enumerate(chunk_batch):
        text_preview = docs[doc_idx].text[:400]
        chunks_text += f"\n[{i+1}] {text_preview}\n"

    prompt = f"""You are a medical relevance judge. Given a patient question and medical text chunks, determine which chunks are relevant to answering the question.

Query: {query}

Chunks:{chunks_text}

Respond with ONLY a JSON array of {len(chunk_batch)} booleans.
Example: [true, false, true]

Answer:"""

    messages = [
        {"role": "system", "content": "You are a precise medical relevance judge. Respond only with a JSON boolean array."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=200, temperature=0.1, do_sample=True, top_p=0.9)

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    try:
        start = response.index("[")
        end = response.index("]") + 1
        results = json.loads(response[start:end])
        return [bool(r) for r in results]
    except:
        return [False] * len(chunk_batch)

print(f"\nAnnotating {len(queries)} queries...")

annotated = []
total_start = time.time()

for qi, q in enumerate(queries):
    query_text = q["query"]
    candidates = candidates_per_query[qi]
    relevant_doc_ids = []

    for batch_start in range(0, len(candidates), LLM_BATCH_SIZE):
        batch = candidates[batch_start:batch_start + LLM_BATCH_SIZE]
        results = judge_batch(query_text, batch)
        for (doc_idx, _), is_relevant in zip(batch, results):
            if is_relevant:
                relevant_doc_ids.append(doc_ids[doc_idx])

    annotated.append({
        "query": query_text,
        "tier": q["tier"],
        "patient_id": q["patient_id"],
        "ground_truth_ids": relevant_doc_ids[:FINAL_TOP_K],
    })

    print(f"  [{qi+1:}/{len(queries)}] {q['tier']}: {len(relevant_doc_ids)} relevant docs")

with open(OUTPUT_PATH, "w") as f:
    json.dump(annotated, f, ensure_ascii=False, indent=2)

print(f"\nDone! Saved to {OUTPUT_PATH}")