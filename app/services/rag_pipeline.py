"""RAG pipeline service — orchestrates retrieval, generation, and safety."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import tiktoken

from app.core.config import cfg
from app.core.messages import MESSAGES
from app.indexing.embedder import embed_texts
from app.indexing.preprocessing import preprocess_query
from app.retrieval.hybrid_fusion import hybrid_retrieval_fusion, deduplicate_results
from app.retrieval.metadata_filter import filter_by_metadata
from app.retrieval.query_expansion import expand_query
from app.retrieval.query_translator import translate_query, is_arabic
from app.retrieval.reranker import rerank as rerank_documents
from app.generation.prompts import build_prompt
from app.generation.groq_client import generate_response
from app.safety.emergency_detector import detect_emergency
from app.safety.content_filter import contains_sensitive_content
from app.safety.judge import judge_answer

logger = logging.getLogger(__name__)

_enc = tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))


def _deduplicate_answer(answer: str, max_repeats: int = 2) -> str:
    if not answer:
        return answer
    lines = answer.split("\n")
    seen: list = []
    counts: Dict[str, int] = {}
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        counts[stripped] = counts.get(stripped, 0) + 1
        if counts[stripped] <= max_repeats:
            seen.append(stripped)
    return "\n".join(seen)


@dataclass
class PipelineResult:
    answer: str
    confidence: float
    sources: List[str]
    grounding_score: float
    status: str
    stage_latencies: Dict[str, float] = field(default_factory=dict)


class RAGPipeline:
    """Stateless RAG pipeline — receives stores at call time."""

    def run(
        self,
        query: str,
        vector_store,
        bm25,
        *,
        role: str = "patient",
        specialty: Optional[str] = None,
        mode: str = "hybrid",
    ) -> PipelineResult:
        stages: Dict[str, float] = {}

        # 1. Emergency detection
        if detect_emergency(query):
            return PipelineResult(
                answer=MESSAGES.EMERGENCY_ESCALATION,
                confidence=1.0,
                sources=[],
                grounding_score=1.0,
                status="emergency_escalation",
                stage_latencies=stages,
            )

        # 2. Preprocessing
        t0 = time.time()
        processed = preprocess_query(query)
        stages["preprocessing"] = round(time.time() - t0, 3)

        # 3. Query expansion
        t0 = time.time()
        expanded = expand_query(processed)
        stages["query_expansion"] = round(time.time() - t0, 3)
        if not expanded:
            return PipelineResult(
                answer="Query expansion failed.",
                confidence=0.0, sources=[], grounding_score=0.0,
                status="expansion_failed", stage_latencies=stages,
            )

        # 4-5. Retrieval
        t0 = time.time()
        dense_results, bm25_results = self._retrieve(
            expanded, processed, vector_store, bm25, mode,
        )
        stages["retrieval"] = round(time.time() - t0, 3)

        if not dense_results and not bm25_results:
            return PipelineResult(
                answer=MESSAGES.NO_RETRIEVAL,
                confidence=0.0, sources=[], grounding_score=0.0,
                status="no_retrieval", stage_latencies=stages,
            )

        # 6. Fusion
        t0 = time.time()
        fused = self._fuse(dense_results, bm25_results, processed, mode)
        stages["fusion"] = round(time.time() - t0, 3)

        # 6b. Score filter
        fused = [d for d in fused if d.get("score", d.get("dense_score", 0.0)) >= 0.15]

        # 7. Metadata filtering
        t0 = time.time()
        filtered = filter_by_metadata(fused, specialty=specialty, language="arabic")
        stages["metadata_filtering"] = round(time.time() - t0, 3)

        # 7b. Reranking
        t0 = time.time()
        candidates = filtered[:cfg.TOP_K_FINAL]
        reranked = rerank_documents(processed, candidates, top_k=cfg.TOP_K_FINAL)
        stages["reranking"] = round(time.time() - t0, 3)

        # 8. Dynamic chunk selection
        top_chunks = [
            c for c in reranked
            if c.get("rerank_score_normalized", c.get("dense_score", 0)) > 0.25
        ][:10]
        if not top_chunks:
            top_chunks = reranked[:3]

        if not top_chunks:
            return PipelineResult(
                answer=MESSAGES.NO_RETRIEVAL,
                confidence=0.0, sources=[], grounding_score=0.0,
                status="no_retrieval", stage_latencies=stages,
            )

        # Enrich metadata
        for doc in top_chunks:
            doc.setdefault("page", doc.get("metadata", {}).get("page"))
            doc.setdefault("source", doc.get("metadata", {}).get("source", "Medical Textbook"))

        # 8b. Context building (token-budgeted)
        t0 = time.time()
        context = self._build_context(top_chunks)
        stages["context_building"] = round(time.time() - t0, 3)

        # 9. Generation
        t0 = time.time()
        try:
            prompt = build_prompt(query, context, role)
            answer = generate_response(prompt)
            answer = _deduplicate_answer(answer)
            stages["generation"] = round(time.time() - t0, 3)
        except Exception as e:
            logger.error("LLM generation failed: %s", type(e).__name__)
            return PipelineResult(
                answer="حدث خطأ أثناء توليد الإجابة. الرجاء المحاولة لاحقاً.",
                confidence=0.0, sources=[], grounding_score=0.0,
                status="llm_failure", stage_latencies=stages,
            )

        # 10. Safety check
        if contains_sensitive_content(answer):
            return PipelineResult(
                answer="لا يمكن عرض هذه المعلومات.",
                confidence=0.0, sources=[], grounding_score=0.0,
                status="blocked_sensitive_content", stage_latencies=stages,
            )

        # 11. Judge (grounding + hallucination)
        t0 = time.time()
        context_texts = [d.get("text", "") for d in top_chunks[:5] if d.get("text")]
        jr = judge_answer(query=query, answer=answer, context_texts=context_texts)
        stages["judge"] = round(time.time() - t0, 3)

        grounding_score = jr.grounding_score
        confidence = round(max(0.0, min(0.95, jr.confidence)), 3)

        if not jr.grounded and grounding_score < 0.3:
            return PipelineResult(
                answer="لا يمكنني تقديم إجابة دقيقة بناءً على المصادر المتاحة.",
                confidence=0.0, sources=[],
                grounding_score=round(grounding_score, 3),
                status="refused_low_grounding", stage_latencies=stages,
            )

        # 12. Citations
        citations = self._build_citations(top_chunks)

        return PipelineResult(
            answer=answer,
            confidence=confidence,
            sources=citations,
            grounding_score=round(grounding_score, 3),
            status="success",
            stage_latencies=stages,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _retrieve(expanded, processed, vector_store, bm25, mode):
        dense_results: list = []
        bm25_results: list = []

        if mode in ("hybrid", "dense"):
            try:
                all_embs = embed_texts(expanded)
                for eq, emb in zip(expanded, all_embs):
                    dense_results.extend(vector_store.search(emb, k=cfg.TOP_K_DENSE))

                # Bilingual boost: if query is Arabic, also search with
                # the English translation to close the cross-lingual gap.
                if is_arabic(processed):
                    en = translate_query(processed)
                    if en and en != processed:
                        en_emb = embed_texts([en])[0]
                        dense_results.extend(
                            vector_store.search(en_emb, k=cfg.TOP_K_DENSE)
                        )
            except Exception as e:
                logger.error("Dense search failed: %s", e)

        if mode in ("hybrid", "bm25") and bm25 is not None:
            try:
                bm25_query = processed
                if is_arabic(processed):
                    bm25_query = translate_query(processed)
                bm25_results.extend(bm25.search(bm25_query, k=cfg.TOP_K_BM25))
            except Exception as e:
                logger.warning("BM25 search failed: %s", e)

        return dense_results, bm25_results

    @staticmethod
    def _fuse(dense_results, bm25_results, processed, mode):
        if mode == "hybrid" and bm25_results and dense_results:
            fused = hybrid_retrieval_fusion(
                dense_results=dense_results,
                bm25_results=bm25_results,
                query=processed,
                top_k=cfg.TOP_K_FINAL,
            )
        elif mode == "bm25":
            fused = deduplicate_results(bm25_results)
        else:
            fused = deduplicate_results(dense_results)

        # Normalize BM25 scores to [0,1]
        raw = [d.get("bm25_score", 0.0) for d in fused if d.get("bm25_score", 0.0) > 0]
        bm25_max = max(raw) if raw else 1.0
        bm25_min = min(raw) if raw else 0.0
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
        for doc in fused:
            doc.setdefault("page", doc.get("metadata", {}).get("page"))
            doc.setdefault("source", doc.get("metadata", {}).get("source", "Medical Textbook"))
            if "dense_score" not in doc:
                raw_bm25 = doc.get("bm25_score", 0.0)
                doc["dense_score"] = (raw_bm25 - bm25_min) / bm25_range if raw_bm25 > 0 else 0.0
        return fused

    @staticmethod
    def _build_context(top_chunks, max_tokens: int = 1500):
        parts: list = []
        token_count = 0
        for doc in top_chunks[:5]:
            text = doc.get("text", "").strip()
            if not text:
                continue
            doc_tokens = _count_tokens(text)
            if token_count + doc_tokens > max_tokens:
                remaining = max_tokens - token_count
                if remaining > 50:
                    tokens = _enc.encode(text)[:remaining]
                    parts.append(_enc.decode(tokens))
                    token_count += remaining
                break
            parts.append(text)
            token_count += doc_tokens
        return "\n\n".join(parts)

    @staticmethod
    def _build_citations(top_chunks):
        seen: set = set()
        unique: list = []
        for doc in top_chunks:
            key = (doc.get("source", "Medical Textbook"), doc.get("page"))
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        citations: list = []
        for doc in unique:
            src = doc.get("source", "Medical Textbook")
            pg = doc.get("page")
            citations.append(f"{src} (Page {pg})" if pg is not None else src)
        return list(dict.fromkeys(citations))


# Module-level singleton
rag_pipeline = RAGPipeline()
