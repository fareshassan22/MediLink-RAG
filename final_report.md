# MediLink RAG — Midterm Report

**Bilingual Medical Retrieval-Augmented Generation System**
**Evaluation Date:** 2026-03-22 | **Hardware:** Multi-GPU RTX A6000 (47 GB VRAM each)

---

## 1. What Is MediLink RAG?

MediLink RAG is a **production-grade, bilingual (Arabic/English) medical question-answering system** built on Retrieval-Augmented Generation. A user asks a medical question in Arabic or English; the system retrieves relevant passages from an indexed medical textbook, generates a grounded answer using an LLM, and validates the answer through multi-layer safety checks before returning it.

**Core problem solved:** Accurate, safe, hallucination-resistant medical Q&A in Arabic — a language underserved by most LLMs — using a constrained knowledge base (Gale Encyclopedia of Medicine, ~759 indexed chunks).

---

## 2. System Architecture & End-to-End Workflow

```
┌─────────────┐
│  User Query  │  (Arabic or English)
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ Emergency Check  │  30+ keywords (AR/EN): chest pain, stroke, نزيف حاد …
│                  │  If matched → immediate escalation, pipeline stops
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│  Preprocessing   │  Arabic normalization (alef/ya/ta-marbuta/diacritics)
│                  │  Punctuation removal, whitespace collapse
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ Query Expansion  │  Medical dictionary lookup → English synonym expansions
│                  │  Output: [original_arabic, "diabetes symptoms", …]
└──────┬───────────┘
       │
       ├───────────────────────────────┐
       ▼                               ▼
┌──────────────┐              ┌─────────────────┐
│    Dense     │              │      BM25       │
│  Retrieval   │              │   Retrieval     │
│  (BGE-M3)   │              │  (BM25-Okapi)   │
│  FAISS, k=20 │              │  Translated EN  │
│  1024-dim    │              │  query, k=20    │
└──────┬───────┘              └────────┬────────┘
       │                               │
       └──────────┬────────────────────┘
                  ▼
       ┌─────────────────────┐
       │   Hybrid Fusion     │
       │  Intent detection   │  (symptoms/causes/treatment/diagnosis/prevention)
       │  Weighted merge     │  Dense 0.75-0.85, BM25 0.15-0.25
       │  Agreement boost    │  Docs in both sets → 1.2× multiplier
       │  Deduplication      │  Jaccard threshold 0.85
       │  Score filter       │  Min keyword overlap 0.1
       └──────┬──────────────┘
              │
              ▼
       ┌─────────────────────┐
       │  Reranking          │  BGE-Reranker-v2-M3 (cross-encoder)
       │  (optional)         │  Re-scores top-10 with query-doc pairs
       └──────┬──────────────┘
              │
              ▼
       ┌─────────────────────┐
       │ Dynamic Selection   │  Score > 0.25 → keep; budget: 1500 tokens
       │ Context Building    │  Top 5-7 chunks concatenated
       └──────┬──────────────┘
              │
              ▼
       ┌─────────────────────┐
       │ Prompt Construction │  System: "You are MediLink, a trusted medical AI"
       │                     │  Rules: answer ONLY from context, max 6 points
       │                     │  Role-adapted (patient vs. doctor)
       └──────┬──────────────┘
              │
              ▼
       ┌─────────────────────┐
       │   LLM Generation    │  Groq API → Llama-3.1-8B-Instant
       │                     │  temp=0.2, max_tokens=1024
       │                     │  Post-process: deduplication, sentence truncation
       └──────┬──────────────┘
              │
              ▼
       ┌─────────────────────┐
       │  Content Filter     │  PII detection (SSN, credit card, phone, email)
       └──────┬──────────────┘
              │
              ▼
       ┌─────────────────────┐
       │   LLM Judge         │  Groq API → Llama-3.1-8B-Instant
       │                     │  Evaluates: grounding, hallucination, confidence
       │                     │  If grounding < 0.3 → REJECT answer
       └──────┬──────────────┘
              │
              ▼
       ┌─────────────────────┐
       │  Response Assembly  │  answer + confidence + sources + grounding_score
       │                     │  + latency breakdown per stage
       └─────────────────────┘
```

**Total latency:** ~4-6 seconds (dominated by two LLM calls: generation ~2s + judge ~2-3s)

---

## 3. What We Built — Component by Component

### 3.1 Indexing Pipeline (Offline)

| Step | Module | What It Does |
|------|--------|-------------|
| PDF Loading | `app/indexing/pdf_loader.py` | Extracts text page-by-page from medical textbook PDF using PyPDF2 |
| Preprocessing | `app/indexing/preprocessing.py` | Arabic normalization (alef variants, ya, ta-marbuta, diacritics removal), punctuation removal, whitespace collapse |
| Arabic Tokenization | `app/indexing/arabic_tokenizer.py` | Custom stemmer (removes 2-3 letter prefixes/suffixes), ~50 Arabic stopwords, bilingual auto-detection |
| Semantic Chunking | `app/indexing/chunker.py` | Paragraph-level splitting, 200-word chunks with 40-word overlap, handles Arabic sentence boundaries |
| Embedding | `app/indexing/embedder.py` | BAAI/bge-m3 (1024-dim, multilingual), normalized L2, batch=64, GPU-accelerated |
| Vector Store | `app/indexing/vector_store.py` | FAISS IndexFlatIP (inner product = cosine for normalized vectors), save/load pickle |
| BM25 Index | `app/indexing/bm25_index.py` | BM25-Okapi (rank_bm25), separate Arabic/English tokenizers, stores corpus+tokens+metadata |

**Result:** 759 indexed chunks from Gale Encyclopedia of Medicine, stored as `vector_store.pkl` + `bm25_index.pkl`.

### 3.2 Retrieval Pipeline (Online, per query)

| Component | Module | Details |
|-----------|--------|---------|
| Query Expansion | `app/retrieval/query_expansion.py` | Arabic→English medical dictionary (~100+ terms); outputs 2-3 query variants |
| Query Translation | `app/retrieval/query_translator.py` | Groq LLM API (Arabic→English) for BM25 path; dictionary fallback if API fails |
| Dense Retrieval | via `app/indexing/vector_store.py` | Multi-query embedding → FAISS search → top-20 per variant |
| BM25 Retrieval | via `app/indexing/bm25_index.py` | English-translated query → BM25-Okapi → top-20 |
| Hybrid Fusion | `app/retrieval/hybrid_fusion.py` | Intent-aware weighted fusion (dense 0.75-0.85 / BM25 0.15-0.25), min-max normalization, agreement boosting (1.2×), Jaccard dedup (0.85), keyword overlap filter (0.1) |
| Reranking | `app/retrieval/reranker.py` | BAAI/bge-reranker-v2-m3 cross-encoder (optional, disabled by default — degraded Arabic Recall@1 to 0.004 in early experiments) |
| Context Compressor | `app/retrieval/context_compressor.py` | Sentence-level relevance ranking by cosine similarity (available but not used in main pipeline) |
| Metadata Filter | `app/retrieval/metadata_filter.py` | Filters by specialty/language if requested; falls back to unfiltered if too restrictive |

### 3.3 Generation Pipeline

| Component | Module | Details |
|-----------|--------|---------|
| Prompt Builder | `app/generation/prompts.py` | Strict template: answer ONLY from context, max 6 points, Arabic output, role-adapted (patient: simple language / doctor: clinical terminology) |
| Groq Client | `app/generation/groq_client.py` | Llama-3.1-8B-Instant via Groq API, temp=0.2, max_tokens=1024, sentence-boundary truncation |
| Answer Dedup | `app/main.py` | Removes lines repeated >2 times (LLM repetition mitigation) |

### 3.4 Safety Pipeline (Multi-Layer)

| Layer | Module | What It Does |
|-------|--------|-------------|
| Emergency Detector | `app/safety/emergency_detector.py` | 30+ Arabic/English emergency keywords ("لا استطيع التنفس", "heart attack", "stroke") → immediate escalation |
| Content Filter | `app/safety/content_filter.py` | PII regex (SSN, credit card, email, phone, national ID) + keyword blocklist → blocks answer |
| LLM Judge | `app/safety/judge.py` | **Primary safety gate.** Groq LLM evaluates grounding (0-1), hallucination risk (0-1), confidence (0-1), flags fabricated claims. Rejects answer if grounding < 0.3. Retry logic with exponential backoff (4 retries). Conservative fallback on API failure. |

**Superseded modules** (kept in codebase but no longer used in main pipeline):
- `grounding.py` — cosine-similarity grounding (replaced by LLM Judge)
- `grounding_checker.py` — alternative embedding-based grounding (replaced by LLM Judge)
- `hallucination_checker.py` — regex + embedding hallucination detection (replaced by LLM Judge)

### 3.5 Confidence Calibration

| Component | Module | Details |
|-----------|--------|---------|
| Calibrator | `app/calibration/calibrator.py` | Logistic Regression on 6 features: grounding_score, retrieval_score, rerank_score, context_length, answer_length, top_similarity. Min 40 samples to train. Outputs ECE, Brier score, accuracy. Fallback heuristic: `0.4×grounding + 0.3×retrieval + 0.2×rerank + 0.1` |
| Data Collector | `app/calibration/data_collector.py` | JSONL-based inference logging, supports human labeling workflow, exports for annotation, imports labels for retraining |

### 3.6 API & Frontend

| Component | Module | Details |
|-----------|--------|---------|
| Main Endpoint | `app/main.py` → `POST /ask` | Input: query (3-500 chars), role, specialty, mode. Output: answer, confidence, sources, grounding_score, latency breakdown |
| Health/Ready | `app/api/routes.py` → `GET /health`, `GET /ready` | Component status checks, k8s-compatible readiness probe |
| Rate Limiter | `app/api/middleware.py` | 60 req/min per IP, sliding window, returns 429 with rate limit headers |
| CORS | `app/api/middleware.py` | Configurable origins via env var, defaults to open |
| Frontend | `frontend/` | Minimal HTML/CSS/JS demo: query form, answer card, sources list, latency breakdown |

### 3.7 Evaluation Pipeline

| Script | Purpose |
|--------|---------|
| `evaluate_retrieval.py` | Benchmarks 4 retrieval modes (dense, bm25, hybrid, hybrid_rerank) on 99-query ground truth. Computes Recall@1/5/10, MRR, nDCG@10, Hit@10. Generates 4 diagnostic plots. |
| `evaluate_plots.py` | End-to-end evaluation: loads Qwen2.5-32B-Instruct (local, bf16, multi-GPU) for generation, runs LLM judge, generates 7 plots (indexing analysis, retrieval quality, recall distribution, grounding histogram, reliability curve, MRR boxplot, generation quality). |
| `run_full_eval.sh` | Orchestrates full evaluation pipeline (retrieval → generation → plots → summary), survives disconnects via nohup. |
| `app/evaluation/metrics.py` | Recall@k, Precision@k, nDCG@k, MRR, ECE, Pearson correlation, grounding/hallucination rates |
| `app/evaluation/ground_truth.py` | Loads/builds ground truth from 99-query JSON, supports keyword + semantic fallback matching |
| `app/evaluation/evaluator.py` | Main evaluator class: runs all 4 modes, computes per-query diagnostics, outputs CSV |

### 3.8 Testing

| Test File | Coverage |
|-----------|----------|
| `tests/test_retrieval.py` | Hybrid fusion, RRF, reranker, query expansion, metadata filtering |
| `tests/test_safety.py` | Emergency detection (Arabic keywords), content filtering, PII regex |
| `tests/test_indexing.py` | Embedder shape/normalization, vector store search, BM25 build |
| `tests/test_calibration.py` | ECE calculation, model training, synthetic data |
| `tests/test_api.py` | `/health`, `/ready`, `/ask` endpoints with mocked components |
| `tests/test_arabic_evaluation.py` | Validates eval dataset structure, fields, category coverage |
| `tests/test_calibration_data.py` | Data collector logging, labeling, record retrieval |

---

## 4. Models Used

| Model | Role | Dimensions | Provider |
|-------|------|-----------|----------|
| BAAI/bge-m3 | Dense embedding (multilingual, Arabic+English) | 1024 | HuggingFace (local GPU) |
| BAAI/bge-reranker-v2-m3 | Cross-encoder reranking (optional) | — | HuggingFace (local GPU) |
| Llama-3.1-8B-Instant | Answer generation + LLM judge + Arabic→English translation | — | Groq API (cloud) |
| Qwen2.5-32B-Instruct | Evaluation-only generation (local, bf16, multi-GPU) | — | HuggingFace (local GPU) |

---

## 5. Evaluation Results

### 5.1 Retrieval Metrics (99 queries, 4 modes)

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@10 | Hit@10 | Zero-Recall Queries |
|------|----------|----------|-----------|-----|---------|--------|---------------------|
| bm25 | 0.06 | 0.20 | 0.29 | 0.41 | 0.27 | 0.66 | 34 |
| **dense** | **0.20** | **0.87** | **0.97** | **0.98** | **0.96** | **0.99** | **1** |
| hybrid | 0.18 | 0.74 | 0.89 | 0.92 | 0.87 | 0.98 | 2 |
| hybrid_rerank | 0.12 | 0.50 | 0.89 | 0.75 | 0.74 | 0.98 | 2 |

**⚠ Ground Truth Caveat:** The current ground truth was auto-generated by running dense retrieval and assigning its top-5 results as "relevant docs" (`remap_ground_truth.py`). This is **circular evaluation** — dense metrics are inflated because the system is recovering its own previous outputs. BM25/hybrid are penalized for returning legitimately different (but possibly correct) documents. These numbers measure **retrieval self-consistency**, not true retrieval quality. Proper evaluation requires human-annotated relevance judgments.

**Observations (relative, not absolute):**
- **BM25 is weakest** (34/99 zero-recall queries) — expected since the corpus is English and BM25 cannot cross the Arabic→English language barrier even with translation.
- **Hybrid fusion** partially recovers BM25 failures but dilutes dense signal — Recall@5 drops from 0.87 → 0.74.
- **Reranking degrades results** — the cross-encoder may not generalize well to Arabic queries against English chunks.
- **Dense appears strongest**, but this is partly an artifact of circular ground truth. True quality requires independent validation.

### 5.2 Generation Evaluation

- **Grounding rate:** Most dense/hybrid answers score ≥ 0.8 grounding.
- **Failure pattern:** Queries where retrieval fails (low recall) propagate to low grounding → judge rejects the answer.
- **BM25-only answers** consistently fail grounding checks due to irrelevant context.
- **hybrid_rerank** occasionally underperforms dense due to the cross-encoder degrading Arabic ranking.

### 5.3 Plots Generated

| # | Plot | File |
|---|------|------|
| 1 | Indexing Analysis (text stats pre/post preprocessing) | `results/plots/1_indexing_analysis.png` |
| 2 | Retrieval Quality (Recall@k, MRR bar chart by mode) | `results/plots/retrieval_quality.png` |
| 3 | Category Heatmap (Recall@10 by category × mode) | `results/plots/retrieval_category_heatmap.png` |
| 4 | MRR Boxplot (per-query variance by mode) | `results/plots/retrieval_mrr_boxplot.png` |

---

## 6. Key Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|----------|-----------|-----------|
| BGE-M3 over monolingual models | Best Arabic+English multilingual embedding model available; 1024-dim captures medical semantics | Larger model, slower embedding than 384-dim alternatives |
| Groq API over local LLM | Fast inference (~2s), no GPU memory needed for generation | Rate-limited (500K tokens/day free tier), external dependency |
| LLM Judge over cosine grounding | LLM can reason about paraphrasing, synonyms, and Arabic medical terminology; cosine similarity gave false negatives on valid Arabic answers | Adds ~2-3s latency, requires second API call, subject to rate limits |
| Dense-primary fusion (0.8/0.2) | BM25 cannot cross the Arabic→English language barrier; dense handles it natively | BM25 contribution is minimal; could simplify to dense-only |
| Disabled reranker by default | BGE-reranker-v2-m3 degraded Arabic Recall@1 to 0.004 in early experiments | Loses potential benefit for English-only queries |
| Overlapping chunks (40-word overlap) | Medical concepts span paragraph boundaries; overlap prevents information loss | ~20% more chunks to index and search |
| Conservative judge fallback | If Groq API fails, assume moderate quality (0.5 grounding) rather than blocking | May pass low-quality answers during API outages |

---

## 7. Configuration Summary

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Embedding model | BAAI/bge-m3 (1024-dim) | Multilingual dense retrieval |
| Reranker | BAAI/bge-reranker-v2-m3 | Cross-encoder (disabled by default) |
| Generation LLM | Llama-3.1-8B-Instant (Groq) | Answer generation + judge + translation |
| Dense weight | 0.80 | Primary retriever |
| BM25 weight | 0.20 | Secondary retriever |
| Top-K dense | 20 | Initial candidates per query variant |
| Top-K BM25 | 20 | Initial BM25 candidates |
| Top-K final | 10 | After fusion/reranking |
| Similarity threshold | 0.25 | Min score for final selection |
| Chunk size | 200 words | Semantic chunking |
| Chunk overlap | 40 words | Context continuity |
| Max context tokens | 1500 | LLM prompt budget |
| Judge grounding reject | < 0.3 | Answer rejection threshold |
| Rate limit | 60 req/min/IP | API protection |
| Random seed | 42 | Reproducibility |

---

## 8. Codebase Status

### Active Components
- `app/main.py` — FastAPI application, main `/ask` endpoint
- `app/indexing/` — Full indexing pipeline (PDF → chunks → embeddings → FAISS + BM25)
- `app/retrieval/` — Dense, BM25, hybrid fusion, translation, expansion, reranking
- `app/generation/` — Prompt building, Groq API client
- `app/safety/judge.py` — LLM-based answer judge (primary safety gate)
- `app/safety/emergency_detector.py` — Emergency keyword detection
- `app/safety/content_filter.py` — PII/sensitive content filtering
- `app/calibration/` — Confidence calibration + data collection
- `app/evaluation/` — Metrics, ground truth management, evaluator class

### Obsolete (safe to delete)
- `app/safety/hallucination_checker.py` — Replaced by LLM Judge
- `app/safety/grounding.py` — Replaced by LLM Judge
- `app/safety/grounding_checker.py` — Replaced by LLM Judge

---

## 9. Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Retrieval metrics (aggregated) | `results/retrieval_metrics.csv` | 4 modes × 7 metrics |
| Retrieval metrics (per-query) | `results/retrieval_per_query.csv` | 99 queries × 4 modes = 396 rows |
| Generation evaluation | `results/generation_eval.csv` | Grounding, confidence, retrieval scores per query/mode |
| Error analysis | `results/error_analysis.json` | Ground truth vs. retrieved doc mismatches |
| Plots | `results/plots/` | 5 diagnostic visualizations |
| Ground truth | `data/eval_ground_truth.json` | 99 curated queries (Arabic+English, 5+ categories, 3 difficulty levels) |
| Vector index | `data/processed/vector_store.pkl` | 759 chunks, FAISS IndexFlatIP |
| BM25 index | `data/processed/bm25_index.pkl` | 759 chunks, BM25-Okapi |

---

## 10. Recommendations & Next Steps

1. **Remove obsolete safety modules** to reduce confusion.
2. **Upgrade Groq API tier** — free tier (500K tokens/day) is insufficient for production or heavy evaluation.
3. **Consider dense-only mode** as default — hybrid fusion dilutes dense signal without meaningful BM25 benefit for Arabic.
4. **Fine-tune reranker on Arabic medical data** or remove it entirely.
5. **Expand ground truth** beyond 99 queries for more robust evaluation.
6. **Add Arabic medical corpus** — current source is English-only; adding Arabic medical literature would improve BM25 and reduce translation dependency.
7. **Implement caching** for repeated queries to reduce API calls and latency.
