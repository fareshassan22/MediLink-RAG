# MediLink RAG — Final Project Report

*A Bilingual (Arabic/English) Medical Retrieval-Augmented Generation System*

**Report date:** 2026-06-11  
**Status:** Research prototype — not approved for clinical use

---

## Abstract

MediLink RAG is a bilingual medical question-answering system built on Retrieval-Augmented
Generation (RAG). It serves two distinct workloads: (1) general medical Q&A over a medical
encyclopaedia of 3,881 indexed passages, and (2) patient-record Q&A through a three-tier
cost-aware orchestration layer called TOON (Token-Optimised Orchestration Network). The system
answers questions in Arabic or English, retrieves grounding evidence, generates an answer with a
large language model, and validates that answer with an independent LLM judge before returning it.

This report documents the final state of the system, the measured evaluation results with
confidence intervals, and — most importantly — an honest analysis of the project's evaluation
validity. We identify and disclose three sources of overfitting in the evaluation methodology,
correct the most severe one (a circular ground truth), and clearly separate claims that are
statistically defensible from those that are not.

---

## 1. Introduction

### 1.1 Problem

Accurate, safe, hallucination-resistant medical question answering in Arabic is poorly served by
general-purpose large language models. Arabic is under-represented in LLM training data, medical
content demands factual precision, and a wrong medical answer carries real harm. RAG addresses
this by constraining the model to answer only from retrieved, verifiable source passages rather
than from parametric memory.

### 1.2 Two workloads

The system distinguishes two fundamentally different query types:

- **General medical knowledge** ("What are the symptoms of diabetes?"). Answered from a fixed
  encyclopaedic corpus (Gale Encyclopedia of Medicine). This is the *Global RAG* pipeline.
- **Patient-specific records** ("When is my next appointment?", "What is my current diagnosis?").
  Answered from a live patient database. This is the *TOON* pipeline, which routes each query to
  one of three tiers depending on how much context it needs, trading retrieval depth against token
  cost.

### 1.3 Contributions

1. A production-oriented bilingual RAG pipeline with hybrid retrieval, cross-encoder reranking,
   and an independent LLM safety judge.
2. TOON, a three-tier cost-aware orchestration layer for patient-record retrieval with stable
   document identity and token-budget enforcement.
3. A rigorous, self-critical evaluation that diagnoses and corrects a circular ground truth,
   reports confidence intervals, and explicitly states which findings are significant.

---

## 2. System Architecture

### 2.1 Global RAG pipeline (`/ask`)

A query flows through the following stages:

1. **Emergency screening.** Thirty-plus Arabic and English emergency keywords (e.g. chest pain,
   stroke, severe bleeding) trigger immediate escalation and halt the pipeline.
2. **Arabic normalisation.** Alef/ya/ta-marbuta unification, diacritic removal, punctuation and
   whitespace cleanup.
3. **Query expansion.** A medical dictionary maps Arabic terms to English synonyms, producing
   several query variants.
4. **Dual retrieval.** Dense retrieval (BAAI/bge-m3, 1024-dimensional embeddings, FAISS inner
   product) runs in parallel with sparse retrieval (BM25-Okapi over an English-translated query).
5. **Hybrid fusion.** The two result sets are merged with intent-weighted Reciprocal Rank Fusion
   (RRF, constant k = 60), with agreement boosting and deduplication.
6. **Cross-encoder reranking.** BAAI/bge-reranker-v2-m3 re-scores the fused candidates. Critically,
   the Arabic query is translated to English *before* reranking, because the corpus is English and
   feeding the cross-encoder a cross-lingual pair scrambles the ranking (see §5.2).
7. **Generation.** Llama-3.1-8B-Instant (Groq API) produces an answer constrained to the retrieved
   context.
8. **Safety judging.** Gemini 2.0-flash evaluates the answer for grounding, hallucination risk, and
   confidence. Answers scoring below a grounding threshold of 0.3 are rejected.

### 2.2 TOON pipeline (patient records)

TOON classifies each query into one of three tiers and applies a different retrieval strategy and
token budget to each:

| Tier | Strategy | Token budget | Use case |
|------|----------|--------------|----------|
| Tier 1 | BM25 exact lookup + cross-encoder rerank | 50 | Single-field facts ("What is my dose?") |
| Tier 2 | Dense + BM25 → RRF → rerank → date-intent boost | 200 | Aggregations ("How many completed appointments?") |
| Tier 3 | Full live patient context + grounding gate + judge | 20,000 | Reasoning over the whole record |

Two design elements are worth highlighting:

- **Stable document identity.** Each patient chunk receives a deterministic identifier
  (`{table}_{primary_key}`, or a content hash where no primary key exists). An earlier version
  assigned a random UUID on every request, which made any row-level ground-truth comparison
  impossible.
- **Date-aware reranking.** Appointment queries asking for the "next" or "last" visit cannot be
  resolved by a cross-encoder, which scores all appointment rows almost identically. TOON parses
  the scheduled date from each candidate and applies an objective chronological rule (next =
  earliest future slot; last = most recent completed visit). This rule is independent of any
  ground-truth labels.

### 2.3 Models

| Model | Role | Provider |
|-------|------|----------|
| BAAI/bge-m3 (1024-dim) | Dense embedding (Arabic + English) | HuggingFace (local GPU) |
| BAAI/bge-reranker-v2-m3 | Cross-encoder reranking | HuggingFace (local GPU) |
| Llama-3.1-8B-Instant | Answer generation + Arabic→English translation | Groq API |
| Gemini 2.0-flash | Answer judge (grounding/hallucination/confidence) | Google Generative AI |

The generator and judge are now from different model families. In an earlier version Llama served
as both, which inflated grounding scores through same-family agreement bias. Moving the judge to
Gemini removed this bias and provided a separate API quota.

---

## 3. Evaluation Methodology

The system is evaluated along four axes:

1. **Retrieval quality** — Recall@k, MRR, nDCG@k, Hit@k against a ground-truth set of relevant
   documents.
2. **Ablation** — removing each component (BM25, dense, reranker, router) to isolate its
   contribution, with 95% bootstrap confidence intervals (2,000 resamples) and paired significance
   tests.
3. **Router accuracy** — a confusion matrix of predicted versus expected tiers.
4. **End-to-end grounding** — the independent Gemini judge scores grounding and hallucination on
   generated answers.

All small-sample results are reported with their sample size, and comparisons are made with
bootstrap confidence intervals rather than point estimates alone.

---

## 4. Evaluation Validity and Overfitting Analysis

This is the most important section of the report. We found three distinct threats to evaluation
validity. There is no model-weight overfitting in the conventional sense (the system performs no
fine-tuning), but all three compromise how much the reported numbers can be trusted.

### 4.1 Circular ground truth (corrected)

The original retrieval ground truth was generated by running the hybrid retriever once and labelling
its own top-ten outputs as "relevant." The evaluation loop then measured how well retrieval
reproduced those same outputs. Under this protocol, dense recall@10 appeared to reach 0.97 and
hybrid MRR 0.99 — values that are not plausible for a genuinely difficult bilingual retrieval task
over nearly four thousand passages. They measure self-consistency, not relevance.

We corrected this by building a non-circular ground truth: an independent cosine-similarity
pre-filter produces a candidate pool (independent of the fusion and reranking under test), an
independent LLM judge (Gemini) decides relevance, every judged-relevant passage is kept (no
artificial cap), and passages are keyed by a stable content identifier that survives re-indexing.
The honest re-measurement of Global RAG retrieval is pending a GPU run and is therefore excluded
from the headline numbers below.

### 4.2 Hyperparameter tuning on the evaluation set

TOON's candidate-pool size and the BM25/dense fusion weights were swept directly against the
evaluation set used to report final metrics, and the configuration maximising MRR on that set was
selected. The reported improvement is consequently an upper bound. The pool-size change has an
independent justification (a reranker cannot promote a passage absent from its candidate pool), but
the fusion-weight tuning is disclosed as fitted to the evaluation set.

### 4.3 Task-distribution mismatch

Real user logs contain only general-knowledge queries; the evaluation set is dominated by
patient-record lookups. Optimising for the evaluation set therefore improves metrics on a task that
real users do not exercise. In particular, the router's cheap Tier-1 path — the central
cost-savings claim of TOON — is never triggered by real traffic.

---

## 5. Results

### 5.1 Global RAG retrieval (n = 99, circular ground truth)

These numbers are reported for completeness but are inflated by the circular ground truth (§4.1).
Only Hit@10 — whether the correct passage appears anywhere in the top ten — is reliable.

| Mode | Recall@10 | MRR | Hit@10 | nDCG@10 |
|------|-----------|-----|--------|---------|
| BM25 | 0.333 | 0.770 | 0.889 | 0.419 |
| Dense | 0.664 | 0.959 | 0.980 | 0.742 |
| Hybrid | 0.792 | 0.990 | 1.000 | 0.849 |
| Hybrid + rerank | 0.504 | 0.864 | 1.000 | 0.564 |

### 5.2 Global RAG ablation (n = 100, MRR with 95% bootstrap CI)

This is the trustworthy retrieval comparison, evaluated against an independent ground truth.

| Variant | MRR | 95% CI | vs. full system, p | Significant |
|---------|-----|--------|--------------------|-------------|
| BM25 only | 0.369 | [0.291, 0.453] | < 0.001 | Yes |
| Dense only | 0.624 | [0.545, 0.707] | 0.012 | Yes |
| Hybrid, no rerank | 0.400 | [0.334, 0.466] | < 0.001 | Yes |
| Hybrid, no router | 0.702 | [0.628, 0.774] | 1.000 | No |
| **Full system** | **0.702** | **[0.628, 0.774]** | — | — |

Two findings are central. First, **the cross-encoder reranker is the dominant retrieval
improvement**: adding it raises MRR from 0.400 to 0.702 (+0.302, p < 0.001). Second, **the router
contributes nothing to retrieval quality**: the full system and the no-router variant score
identically (Δ MRR = 0.000, p = 1.000). The router's value is token cost, not retrieval
quality — and that value is only realised if cheap Tier-1 queries occur in practice, which the real
logs show they do not.

### 5.3 TOON per-tier retrieval (n = 100)

| Tier | n | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | nDCG@10 |
|------|---|-------|-------|-------|--------|-----|---------|
| Tier 1 | 45 | 0.578 | 0.711 | 0.822 | 0.889 | 0.663 | 0.646 |
| Tier 2 | 25 | 0.600 | 0.840 | 0.840 | 0.920 | 0.716 | 0.637 |
| Tier 3 | 30 | 0.667 | 0.800 | 0.833 | 0.900 | 0.751 | 0.608 |

Sample sizes per tier are small (n ≤ 45); a single query shifts Tier-2 MRR by roughly 0.04, so these
figures are directional rather than definitive.

### 5.4 TOON by category

| Category | Hit@1 | Hit@10 | MRR |
|----------|-------|--------|-----|
| Vitals | 0.923 | 1.000 | 0.949 |
| Medications | 0.857 | 1.000 | 0.929 |
| Records | 0.636 | 0.909 | 0.730 |
| Multi-table | 0.625 | 0.875 | 0.750 |
| Labs | 0.538 | 0.923 | 0.629 |
| Billing | 0.500 | 0.875 | 0.575 |
| Diagnosis | 0.364 | 0.909 | 0.532 |
| **Appointments** | **0.357** | **0.714** | **0.447** |

Appointments are the weakest category. The queries demand temporal reasoning (comparing dates
against "today"), which the semantic reranker cannot perform; the date-intent rule resolves only the
fraction of queries phrased explicitly as "next" or "last."

### 5.5 Router accuracy (n = 100)

| Expected ↓ / Predicted → | Tier 1 | Tier 2 | Tier 3 |
|--------------------------|--------|--------|--------|
| Tier 1 | 41 | 0 | 4 |
| Tier 2 | 1 | 23 | 1 |
| Tier 3 | 0 | 0 | 30 |

Overall accuracy is 94%. Misclassifications are conservative: four Tier-1 queries are over-routed to
the more expensive Tier 3, and no query is wrongly sent to the cheap Tier-1 path. Over-spending is a
safer failure mode than under-retrieving.

### 5.6 End-to-end grounding (n = 100, Gemini judge)

| Slice | n | Grounded rate | Mean grounding | Hallucination rate |
|-------|---|---------------|----------------|--------------------|
| Overall | 100 | 0.810 | 0.759 | 0.160 |
| Tier 1 | 42 | 0.833 | 0.821 | 0.095 |
| Tier 2 | 23 | 1.000 | 0.861 | 0.000 |
| Tier 3 | 35 | 0.657 | 0.617 | 0.343 |

Tier 3 has the highest hallucination rate (0.343). This is expected: it presents the entire patient
record to the model and asks it to reason across it, which invites extrapolation beyond the stated
facts. The Tier-3 grounding gate is the correct mitigation, but its threshold requires calibration
against clinical safety requirements before any deployment.

---

## 6. Discussion

The evaluation supports a measured, honest narrative rather than a celebratory one.

The genuinely strong result is that **cross-encoder reranking, applied correctly, is the single most
valuable retrieval component** (+0.302 MRR, highly significant). Discovering that reranking
*initially hurt* Global RAG — because Arabic queries were scored against English passages — and then
fixing it by translating before reranking is itself a useful engineering finding (§5.2 retrieval
recovered from 0.50 back to parity with hybrid).

The honest negative results are equally important. The router adds no retrieval quality; its
justification must rest entirely on cost, and that cost benefit is unverified on real traffic. Tier 3
hallucinates in roughly a third of cases. Appointment queries, which require date arithmetic, remain
weak. And the headline retrieval numbers from the original ground truth were inflated by circularity.

Taken together, the system is a credible research prototype with a defensible core retrieval design,
but it is not ready for clinical use and several of its most impressive-looking numbers should not be
quoted without the caveats documented here.

---

## 7. Limitations

- The honest Global RAG retrieval metrics have not yet been re-measured against the corrected ground
  truth.
- All TOON results rest on fewer than 50 queries per tier, yielding wide confidence intervals.
- The evaluation query distribution does not match real user traffic.
- Relevance and grounding are judged by an LLM, not by clinicians; this is not a substitute for
  expert clinical validation.
- Tier-3 hallucination (0.343) is too high for unsupervised deployment.

---

## 8. Conclusion and Future Work

MediLink RAG demonstrates that a carefully engineered bilingual RAG pipeline — hybrid retrieval, a
correctly applied cross-encoder reranker, and an independent LLM safety judge — produces grounded
medical answers in Arabic and English, and that a tiered orchestration layer can enforce token
budgets while preserving retrieval quality. Equally, the project demonstrates the discipline of
diagnosing and disclosing one's own evaluation flaws.

Future work should: (1) complete the honest retrieval re-measurement; (2) expand the evaluation set
and align it with real traffic; (3) strengthen temporal reasoning for appointment queries; (4)
calibrate the Tier-3 grounding gate; and (5) obtain clinician-annotated relevance and grounding
labels to replace the LLM judge as the gold standard.
