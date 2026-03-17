from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import os
import time

from app.utils.seed import set_seed, DEFAULT_SEED

set_seed(DEFAULT_SEED)

from app.api.middleware import RateLimitMiddleware

# Indexing
from app.indexing.embedder import embed_texts
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index
from app.indexing.preprocessing import preprocess_query

# Retrieval
from app.retrieval.hybrid_fusion import hybrid_retrieval_fusion, deduplicate_results
from app.retrieval.metadata_filter import filter_by_metadata
from app.retrieval.query_expansion import expand_query
from app.retrieval.query_translator import translate_query, is_arabic
from app.retrieval.reranker import rerank as rerank_documents

# Generation
from app.generation.prompts import build_prompt
from app.generation.groq_client import generate_response

# Safety
from app.safety.emergency_detector import detect_emergency
from app.safety.content_filter import contains_sensitive_content
from app.safety.grounding_checker import verify_grounding

# Core
from app.core.messages import MESSAGES
from app.core.state import get_state, set_vector_store, set_bm25, set_ready

# Utils
from app.utils.logger import log_request


def deduplicate_answer(answer: str, max_repeats: int = 2) -> str:
    """Remove repeated phrases from LLM response."""
    if not answer:
        return answer

    lines = answer.split("\n")
    seen_lines: list = []
    repeat_count: Dict[str, int] = {}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        repeat_count[stripped] = repeat_count.get(stripped, 0) + 1

        if repeat_count[stripped] <= max_repeats:
            seen_lines.append(stripped)

    return "\n".join(seen_lines)


# ==========================
# App Setup
# ==========================

app = FastAPI(title="MediLink API")

# CORS — restrict in production via MEDILINK_CORS_ORIGINS env var
_allowed_origins = os.getenv("MEDILINK_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)

from app.api.routes import router as api_router

app.include_router(api_router, prefix="/api")


# ==========================
# Startup: Load Indexes
# ==========================


@app.on_event("startup")
async def load_models():
    from app.indexing.index_manager import index_manager

    needs_rebuild, reason = index_manager.needs_rebuild(min_documents=100)
    if needs_rebuild:
        print(f"[WARN] Index health check: {reason}")
        print("[INFO] Please rebuild indexes using the indexing pipeline")

    vs = VectorStore(dim=1024)
    try:
        vs.load("data/processed")
        if len(vs.documents) == 0:
            raise RuntimeError("Vector store is empty - please rebuild index")
    except Exception as e:
        print(f"[ERROR] Vector store load failed: {e}")
        raise RuntimeError(f"Failed to load vector store: {e}")

    bm25 = None
    try:
        bm25 = BM25Index.load("data/processed")
    except Exception as e:
        print(f"[WARN] BM25 index load failed: {e}")

    set_vector_store(vs)
    set_bm25(bm25)
    set_ready(True)
    print(f"[INFO] Indexes loaded - {len(vs.documents)} documents")


# ==========================
# Request / Response Models
# ==========================


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    role: str = Field(default="patient")
    specialty: Optional[str] = None
    mode: str = Field(default="hybrid", description="Retrieval mode: hybrid, dense, or bm25")

    @validator("query")
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError("Query must not be empty")
        return v.strip()

    @validator("mode")
    def validate_mode(cls, v):
        allowed = {"hybrid", "dense", "bm25"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v


class ResponseModel(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    grounding_score: float
    latency_seconds: float
    status: str
    stage_latencies: Optional[Dict[str, float]] = None


# ==========================
# Main RAG Endpoint
# ==========================


@app.post("/ask", response_model=ResponseModel)
def ask(request: QueryRequest):
    start_time = time.time()
    query = request.query
    stage_latencies: Dict[str, float] = {}

    state = get_state()
    state.ensure_ready()
    vector_store = state.vector_store
    bm25 = state.bm25

    try:
        # -- 1. Emergency Detection --
        if detect_emergency(query):
            latency = round(time.time() - start_time, 3)
            return ResponseModel(
                answer=MESSAGES.EMERGENCY_ESCALATION,
                confidence=1.0,
                sources=[],
                grounding_score=1.0,
                latency_seconds=latency,
                status="emergency_escalation",
                stage_latencies=stage_latencies,
            )

        # -- 2. Query Preprocessing --
        t0 = time.time()
        processed_query = preprocess_query(query)
        stage_latencies["preprocessing"] = round(time.time() - t0, 3)

        # -- 3. Query Expansion --
        t0 = time.time()
        expanded_queries = expand_query(processed_query)
        stage_latencies["query_expansion"] = round(time.time() - t0, 3)

        if not expanded_queries:
            raise HTTPException(status_code=400, detail="Query expansion failed")

        print(f"[DEBUG] Original: {query}")
        print(f"[DEBUG] Expanded: {expanded_queries}")

        retrieval_mode = request.mode
        print(f"[DEBUG] Retrieval mode: {retrieval_mode}")

        # -- 4. Dense Retrieval (multi-query) --
        t0 = time.time()
        dense_results: list = []

        if retrieval_mode in ("hybrid", "dense"):
            try:
                all_embeddings = embed_texts(expanded_queries)
                for eq, emb in zip(expanded_queries, all_embeddings):
                    hits = vector_store.search(emb, k=10)
                    dense_results.extend(hits)
                    print(f"[DEBUG] Dense '{eq[:40]}': {len(hits)} hits")
            except Exception as e:
                print(f"[ERROR] Dense search failed: {e}")

        stage_latencies["dense_retrieval"] = round(time.time() - t0, 3)
        print(f"[DEBUG] Total dense results: {len(dense_results)}")

        # -- 5. BM25 Retrieval (translate Arabic → English for English corpus) --
        t0 = time.time()
        bm25_results: list = []

        if retrieval_mode in ("hybrid", "bm25"):
            if bm25 is not None:
                try:
                    bm25_query = processed_query
                    if is_arabic(processed_query):
                        bm25_query = translate_query(processed_query)
                        print(f"[DEBUG] BM25 translated query: {bm25_query}")
                    bm25_hits = bm25.search(bm25_query, k=10)
                    bm25_results.extend(bm25_hits)
                    print(f"[DEBUG] BM25 results: {len(bm25_hits)} hits")
                except Exception as e:
                    print(f"[WARN] BM25 search failed: {e}")
            else:
                print("[WARN] BM25 index not loaded — running dense-only")

        stage_latencies["bm25_retrieval"] = round(time.time() - t0, 3)

        if not dense_results and not bm25_results:
            latency = round(time.time() - start_time, 3)
            return ResponseModel(
                answer=MESSAGES.NO_RETRIEVAL,
                confidence=0.0,
                sources=[],
                grounding_score=0.0,
                latency_seconds=latency,
                status="no_retrieval",
                stage_latencies=stage_latencies,
            )

        # -- 6. Weighted Hybrid Fusion --
        t0 = time.time()
        if retrieval_mode == "hybrid" and bm25_results and dense_results:
            fused = hybrid_retrieval_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
                query=processed_query,
                top_k=10,
            )
            print(f"[DEBUG] After hybrid fusion: {len(fused)}")
        elif retrieval_mode == "bm25":
            fused = deduplicate_results(bm25_results)
            print(f"[DEBUG] BM25-only after dedup: {len(fused)}")
        else:
            # Dense-only fallback with deduplication
            fused = deduplicate_results(dense_results)
            print(f"[DEBUG] Dense-only after dedup: {len(fused)}")
        stage_latencies["fusion"] = round(time.time() - t0, 3)

        # Normalize BM25 scores to [0,1] via min-max
        bm25_raw_scores = [d.get("bm25_score", 0.0) for d in fused if d.get("bm25_score", 0.0) > 0]
        bm25_max = max(bm25_raw_scores) if bm25_raw_scores else 1.0
        bm25_min = min(bm25_raw_scores) if bm25_raw_scores else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
        for doc in fused:
            doc.setdefault("page", doc.get("metadata", {}).get("page"))
            doc.setdefault("source", doc.get("metadata", {}).get("source", "Medical Textbook"))
            # Keep real dense_score if present; normalize BM25 score as fallback
            if "dense_score" not in doc:
                raw_bm25 = doc.get("bm25_score", 0.0)
                doc["dense_score"] = (raw_bm25 - bm25_min) / bm25_range if raw_bm25 > 0 else 0.0

        # -- 6b. Drop low-score chunks --
        min_score = 0.15
        before_filter = len(fused)
        fused = [d for d in fused if d.get("score", d.get("dense_score", 0.0)) >= min_score]
        print(f"[DEBUG] Score filter: {before_filter} -> {len(fused)} (min={min_score})")

        # -- 7. Metadata Filtering --
        t0 = time.time()
        filtered = filter_by_metadata(fused, specialty=request.specialty, language="arabic")
        stage_latencies["metadata_filtering"] = round(time.time() - t0, 3)
        print(f"[DEBUG] After filter: {len(filtered)}")

        # -- 7b. Reranking (cross-encoder on top 10) --
        t0 = time.time()
        candidates = filtered[:10]
        reranked = rerank_documents(processed_query, candidates, top_k=10)
        stage_latencies["reranking"] = round(time.time() - t0, 3)
        print(f"[DEBUG] Reranked {len(candidates)} candidates")

        # -- 8. Dynamic Chunk Selection (score > 0.5, max 5) --
        top_chunks = [c for c in reranked if c.get("rerank_score_normalized", c.get("dense_score", 0)) > 0.5][:5]
        if not top_chunks:
            top_chunks = reranked[:1]  # fallback: always keep at least 1
        print(f"[DEBUG] Dynamic selection: {len(top_chunks)} chunks (from {len(reranked)} reranked)")

        if not top_chunks:
            latency = round(time.time() - start_time, 3)
            return ResponseModel(
                answer=MESSAGES.NO_RETRIEVAL,
                confidence=0.0,
                sources=[],
                grounding_score=0.0,
                latency_seconds=latency,
                status="no_retrieval",
                stage_latencies=stage_latencies,
            )

        for i, doc in enumerate(top_chunks):
            doc.setdefault("page", doc.get("metadata", {}).get("page"))
            doc.setdefault("source", doc.get("metadata", {}).get("source", "Medical Textbook"))
            rs = doc.get('rerank_score_normalized', doc.get('dense_score', 0))
            print(f"[DEBUG] Chunk {i}: page={doc.get('page')}, score={rs:.3f}")

        # -- 8. Build Context --
        t0 = time.time()
        context_parts: list = []
        token_count = 0
        max_context_tokens = 1200

        for doc in top_chunks[:5]:
            text = doc.get("text", "").strip()
            if not text:
                continue
            words = text.split()
            if token_count + len(words) > max_context_tokens:
                remaining = max_context_tokens - token_count
                if remaining > 50:
                    context_parts.append(" ".join(words[:remaining]))
                break
            context_parts.append(text)
            token_count += len(words)

        stage_latencies["context_building"] = round(time.time() - t0, 3)
        context = "\n\n".join(context_parts)
        print(f"[DEBUG] Context chunks: {len(context_parts)}, tokens: {token_count}")

        # -- 9. LLM Generation --
        t0 = time.time()
        try:
            prompt = build_prompt(query, context, request.role)
            print(f"[DEBUG] Prompt length: {len(prompt)}")
            answer = generate_response(prompt)
            answer = deduplicate_answer(answer)
            print(f"[DEBUG] Answer: {answer[:100]}...")
            stage_latencies["generation"] = round(time.time() - t0, 3)
        except Exception as e:
            print(f"[ERROR] LLM generation failed: {e}")
            import traceback
            traceback.print_exc()
            latency = round(time.time() - start_time, 3)
            return ResponseModel(
                answer="حدث خطأ أثناء توليد الإجابة. الرجاء المحاولة لاحقاً.",
                confidence=0.0,
                sources=[],
                grounding_score=0.0,
                latency_seconds=latency,
                status="llm_failure",
                stage_latencies=stage_latencies,
            )

        # -- 10. Grounding Verification --
        t0 = time.time()
        context_texts = [doc.get("text", "") for doc in top_chunks[:5] if doc.get("text")]
        grounded, grounding_score = verify_grounding(
            answer=answer, context_chunks=context_texts, threshold=0.50
        )
        stage_latencies["grounding"] = round(time.time() - t0, 3)
        print(f"[DEBUG] Grounding: grounded={grounded}, score={round(grounding_score, 3)}")

        if not grounded:
            latency = round(time.time() - start_time, 3)
            print(f"[WARN] Answer rejected - low grounding score: {grounding_score}")
            return ResponseModel(
                answer="لا يمكنني تقديم إجابة دقيقة بناءً على المصادر المتاحة.",
                confidence=0.0,
                sources=[],
                grounding_score=round(grounding_score, 3),
                latency_seconds=latency,
                status="refused_low_grounding",
                stage_latencies=stage_latencies,
            )

        # -- 11. Safety Check --
        if contains_sensitive_content(answer):
            latency = round(time.time() - start_time, 3)
            print("[WARN] Answer blocked - sensitive content")
            return ResponseModel(
                answer="لا يمكن عرض هذه المعلومات.",
                confidence=0.0,
                sources=[],
                grounding_score=round(grounding_score, 3),
                latency_seconds=latency,
                status="blocked_sensitive_content",
                stage_latencies=stage_latencies,
            )

        # -- 12. Confidence Scoring --
        # Use best chunk score (top-1) so more chunks never penalizes confidence
        retrieval_scores = []
        for d in top_chunks:
            score = d.get("rerank_score_normalized", d.get("dense_score", 0.0))
            retrieval_scores.append(max(0.0, min(1.0, score)))

        retrieval_score = max(retrieval_scores) if retrieval_scores else 0.0
        # Evidence bonus: +0.03 per extra chunk scoring > 0.5
        evidence_bonus = sum(1 for s in retrieval_scores[1:] if s > 0.5) * 0.03
        retrieval_score = min(1.0, retrieval_score + evidence_bonus)

        # BM25-only uses translated queries — inherently less precise
        if retrieval_mode == "bm25":
            confidence = 0.50 * grounding_score + 0.35 * retrieval_score
        else:
            confidence = 0.55 * grounding_score + 0.45 * retrieval_score
        confidence = round(max(0.0, min(0.95, confidence)), 3)

        print(f"[DEBUG] Grounding={grounding_score:.3f}, Retrieval={retrieval_score:.3f} (bonus={evidence_bonus:.2f}), Confidence={confidence}")

        # -- 13. Citations --
        seen_pages: set = set()
        unique_chunks: list = []
        for doc in top_chunks:
            key = (doc.get("source", "Medical Textbook"), doc.get("page"))
            if key not in seen_pages:
                seen_pages.add(key)
                unique_chunks.append(doc)

        citations: list = []
        for doc in unique_chunks:
            source = doc.get("source", "Medical Textbook")
            page = doc.get("page")
            citations.append(f"{source} (Page {page})" if page is not None else source)

        citations = list(dict.fromkeys(citations))
        latency = round(time.time() - start_time, 3)

        print(f"[DEBUG] Sources: {citations}")

        response = ResponseModel(
            answer=answer,
            confidence=confidence,
            sources=citations,
            grounding_score=round(grounding_score, 3),
            latency_seconds=latency,
            status="success",
            stage_latencies=stage_latencies,
        )

        log_request(
            {
                "query": query,
                "expanded_queries": expanded_queries,
                "grounding_score": grounding_score,
                "confidence": confidence,
                "latency": latency,
                "status": "success",
            }
        )

        return response

    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="internal_server_error")
