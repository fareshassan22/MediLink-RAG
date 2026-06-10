#!/usr/bin/env python3
"""Build ROW-LEVEL ground truth for TOON retrieval evaluation.

Unlike annotate_toon_ground_truth.py (which judges the GLOBAL textbook corpus),
this judges TOON's OWN per-patient rows. For each TOON test query it asks the
Qwen judge which of the patient's indexed rows are relevant, and records their
STABLE doc_ids (table_pk / table_hash). The result lets evaluate_toon.py compute
real Recall@K / Precision@K / NDCG@K / MRR for TOON's Tier-1 (BM25) and Tier-2
(hybrid RRF) retrieval.

Patients have ~26 rows, so we judge ALL rows per query (no embedding prefilter).

Run on GPU 0:
    CUDA_VISIBLE_DEVICES=0 python3 build_toon_rowlevel_gt.py

Output: data/toon_rowlevel_ground_truth.json
    [{"query", "tier", "patient_id", "relevant_ids": [doc_id, ...]}]
"""

import json
import os
import time

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from app.retrieval import toon
from tests.test_rag_queries import TOON_TEST_QUERIES

MODEL_NAME = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-32B-Instruct")
OUTPUT_PATH = "data/toon_rowlevel_ground_truth.json"
JUDGE_BATCH = 13  # rows per judge call

# Optional multi-patient query set (data/toon_multipatient_queries.json).
# Set TOON_QUERY_SET to a JSON file of [{"query","tier","patient_id"}] to judge
# across many patients instead of the single-patient TOON_TEST_QUERIES.
_QUERY_SET = os.environ.get("TOON_QUERY_SET", "")

# ─── Collect queries ──────────────────────────────────────────────────────────
queries = []
if _QUERY_SET and os.path.exists(_QUERY_SET):
    with open(_QUERY_SET, "r", encoding="utf-8") as f:
        for item in json.load(f):
            queries.append(
                {"query": item["query"], "tier": item["tier"], "patient_id": int(item["patient_id"])}
            )
    OUTPUT_PATH = os.environ.get("TOON_GT_OUTPUT", "data/toon_rowlevel_ground_truth_multipatient.json")
    print(f"{len(queries)} queries from {_QUERY_SET}")
else:
    for tier, tier_data in TOON_TEST_QUERIES.items():
        for q in tier_data["queries"]["questions"]:
            queries.append(
                {"query": q, "tier": tier, "patient_id": tier_data["queries"]["patient_id"]}
            )
    print(f"{len(queries)} TOON queries")

# ─── Build / load each patient's row set (stable doc_ids) ─────────────────────
patient_ids = sorted({q["patient_id"] for q in queries})
patient_rows = {}  # pid -> list of {"doc_id", "text"}
for pid in patient_ids:
    chunks = toon.fetch_all_chunks(pid)
    patient_rows[pid] = [
        {"doc_id": c["metadata"]["doc_id"], "text": c["text"]} for c in chunks
    ]
    print(f"  patient {pid}: {len(patient_rows[pid])} rows")
    # Ensure the live index exists with the same stable ids (for the evaluator)
    vs, bm25 = toon.load_patient_index(pid)
    if not vs and not bm25:
        toon.index_patient(pid)

# ─── Load judge ───────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME} (4-bit)...")
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, quantization_config=quant, device_map="auto", trust_remote_code=True
)
model.eval()
print("  Model loaded")


def judge_batch(query: str, rows_batch: list) -> list:
    chunks_text = ""
    for i, row in enumerate(rows_batch):
        chunks_text += f"\n[{i+1}] {row['text'][:400]}\n"

    prompt = f"""You are a medical relevance judge. Given a patient question and rows from that patient's medical record, decide which rows are relevant to answering the question.

Query: {query}

Rows:{chunks_text}

Respond with ONLY a JSON array of {len(rows_batch)} booleans.
Example: [true, false, true]

Answer:"""

    messages = [
        {"role": "system", "content": "You are a precise medical relevance judge. Respond only with a JSON boolean array."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=120, temperature=0.1, do_sample=True, top_p=0.9)
    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        start = response.index("[")
        end = response.index("]") + 1
        results = json.loads(response[start:end])
        out = [bool(r) for r in results]
        if len(out) != len(rows_batch):
            out = (out + [False] * len(rows_batch))[: len(rows_batch)]
        return out
    except Exception:
        return [False] * len(rows_batch)


# ─── Annotate ─────────────────────────────────────────────────────────────────
print(f"\nAnnotating {len(queries)} queries...")
annotated = []
t0 = time.time()
for qi, q in enumerate(queries):
    rows = patient_rows[q["patient_id"]]
    relevant_ids = []
    for b in range(0, len(rows), JUDGE_BATCH):
        batch = rows[b : b + JUDGE_BATCH]
        flags = judge_batch(q["query"], batch)
        for row, is_rel in zip(batch, flags):
            if is_rel:
                relevant_ids.append(row["doc_id"])
    annotated.append(
        {
            "query": q["query"],
            "tier": q["tier"],
            "patient_id": q["patient_id"],
            "category": q.get("category"),
            "relevant_ids": relevant_ids,
        }
    )
    print(f"  [{qi+1}/{len(queries)}] {q['tier']}: {len(relevant_ids)} relevant rows")

os.makedirs("data", exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(annotated, f, ensure_ascii=False, indent=2)

n_empty = sum(1 for a in annotated if not a["relevant_ids"])
print(f"\nDone in {time.time()-t0:.0f}s → {OUTPUT_PATH}")
print(f"  queries with >=1 relevant row: {len(annotated)-n_empty}/{len(annotated)}")
