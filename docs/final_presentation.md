---
marp: true
title: "MediLink RAG — Final Discussion"
theme: default
paginate: true
size: 16:9
---

<!-- _paginate: false -->

# MediLink RAG

## A Bilingual (Arabic / English) Medical RAG System

**Final Project Discussion** · 2026-06-11

*Retrieval-Augmented Generation · Hybrid Retrieval · Cost-Aware Orchestration · Honest Evaluation*

---

## The Problem

- Medical Q&A in **Arabic** is poorly served by general LLMs
- Medical answers demand **factual precision** — a wrong answer causes harm
- LLMs **hallucinate** from parametric memory

**Our approach:** constrain the model to answer **only from retrieved, verifiable
passages** — Retrieval-Augmented Generation.

---

## Two Workloads, One System

| Workload | Source | Pipeline |
|----------|--------|----------|
| **General knowledge**<br>"symptoms of diabetes?" | Medical encyclopaedia<br>(3,881 passages) | **Global RAG** |
| **Patient records**<br>"when is my next appointment?" | Live patient database | **TOON**<br>(3-tier, cost-aware) |

---

## Global RAG Pipeline

```
Query (AR/EN)
  → Emergency screening (30+ keywords)
  → Arabic normalisation
  → Query expansion (medical dictionary)
  → Dense (BGE-M3, FAISS)  +  BM25 (translated query)
  → Hybrid RRF fusion (intent-weighted, k=60)
  → Cross-encoder rerank (translate → English first!)
  → Generation (Llama-3.1-8B, Groq)
  → Judge (Gemini 2.0-flash): grounding / hallucination
```

---

## TOON — Token-Optimised Orchestration

| Tier | Strategy | Budget | Use case |
|------|----------|--------|----------|
| **T1** | BM25 + rerank | 50 tok | Single-field facts |
| **T2** | Dense+BM25 → RRF → rerank → date boost | 200 tok | Aggregations |
| **T3** | Full patient context + grounding gate | 20,000 tok | Reasoning |

**Key fixes:** stable document IDs (was random UUID), date-aware rerank for
appointments, token-budget enforcement.

---

## Models

| Model | Role |
|-------|------|
| **BAAI/bge-m3** (1024-dim) | Dense embedding (AR + EN) |
| **BAAI/bge-reranker-v2-m3** | Cross-encoder reranking |
| **Llama-3.1-8B-Instant** (Groq) | Generation + translation |
| **Gemini 2.0-flash** (Google) | Independent judge |

Generator ≠ Judge → removes **same-family agreement bias**.

---

# The Honest Part

## We found 3 overfitting / validity threats

*No model-weight overfitting (no fine-tuning) — but the evaluation itself was compromised.*

---

## Threat 1 — Circular Ground Truth (severe)

The original ground truth = **the retriever's own top-10 outputs**.

> Measuring against it asks *"can you reproduce yourself?"* — not *"are you correct?"*

- Dense recall@10 looked like **0.97**, hybrid MRR **0.99** — implausible
- **Fixed:** new GT builder — independent cosine pool + **Gemini judge**, stable
  content IDs, no artificial cap
- Honest re-measurement **pending GPU run** → excluded from headline numbers

---

## Threats 2 & 3

**Threat 2 — Tuning on the eval set**
- Pool-size and fusion-weight swept against the *same* set used to report results
- Reported gains are an **upper bound**; disclosed as fitted

**Threat 3 — Distribution mismatch**
- Real traffic = 100% general-knowledge queries
- Eval set = 60%+ patient-record lookups
- The router's cheap T1 path **fires on 0% of real traffic**

---

## Result — Retrieval Ablation (n=100, 95% CI)

| Variant | MRR | 95% CI | Significant? |
|---------|-----|--------|--------------|
| BM25 only | 0.369 | [0.291, 0.453] | ✓ |
| Dense only | 0.624 | [0.545, 0.707] | ✓ |
| Hybrid, no rerank | 0.400 | [0.334, 0.466] | ✓ |
| Hybrid, no router | 0.702 | [0.628, 0.774] | ✗ |
| **Full system** | **0.702** | **[0.628, 0.774]** | — |

---

## Two Central Findings

✅ **Cross-encoder rerank is the dominant win**
`0.400 → 0.702 MRR  (+0.302, p < 0.001)`

❌ **The router adds ZERO retrieval quality**
`full = no-router  (Δ = 0.000, p = 1.000)`
→ its value is **cost**, and that cost benefit is unverified on real traffic.

---

## Result — TOON Per-Tier (n=100)

| Tier | n | Hit@1 | Hit@10 | MRR |
|------|---|-------|--------|-----|
| T1 | 45 | 0.578 | 0.889 | 0.663 |
| T2 | 25 | 0.600 | 0.920 | 0.716 |
| T3 | 30 | 0.667 | 0.900 | 0.751 |

*Small n per tier — one query ≈ 0.04 MRR. Directional, not definitive.*

---

## Result — TOON by Category

| Strong | Hit@1 | | Weak | Hit@1 |
|--------|-------|---|------|-------|
| Vitals | 0.923 | | Billing | 0.500 |
| Meds | 0.857 | | Diagnosis | 0.364 |
| Records | 0.636 | | **Appointments** | **0.357** |

**Appointments are weakest** — they need date arithmetic the reranker can't do.

---

## Result — Router & Grounding

**Router accuracy: 94%** (conservative errors — over-spend, never under-retrieve)

| Slice | Grounded | Hallucination |
|-------|----------|---------------|
| Overall | 0.810 | 0.160 |
| T2 | 1.000 | 0.000 |
| **T3** | **0.657** | **0.343** |

T3 reasons over the full record → highest hallucination → grounding gate needed.

---

## What We Can / Cannot Claim

| Claim | Verdict |
|-------|---------|
| Reranker is the main retrieval win | ✅ measured, significant |
| Router accuracy 94% | ✅ (eval distribution only) |
| Dense recall@10 = 0.97 | ❌ circular-GT artifact |
| Router improves retrieval quality | ❌ Δ = 0.000 |
| System is clinically safe | ❌ no clinician validation |

---

## Limitations

- Honest Global-RAG retrieval metrics **not yet re-measured**
- All TOON results: **n < 50 per tier** → wide CIs
- Eval distribution ≠ real traffic
- Relevance & grounding judged by an **LLM, not clinicians**
- T3 hallucination **0.343** — too high for deployment

---

## Conclusion

- A credible, well-engineered bilingual RAG prototype
- **Strong:** correctly-applied cross-encoder reranking (+0.302 MRR)
- **Honest negatives:** router cost-benefit unverified, T3 hallucination, circular GT
- The real contribution is **diagnosing and disclosing our own evaluation flaws**

> Not clinically ready — but every number is defensible or clearly caveated.

---

## Future Work

1. Complete the **honest retrieval re-measurement**
2. **Expand** the eval set and align it with real traffic
3. Strengthen **temporal reasoning** (appointments)
4. **Calibrate** the Tier-3 grounding gate
5. Obtain **clinician-annotated** gold labels

---

<!-- _paginate: false -->

# Thank You

### Questions & Discussion

*MediLink RAG · Bilingual Medical Retrieval-Augmented Generation*
