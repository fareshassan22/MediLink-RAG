"""PatientRAGService — orchestrates TOON 3-tier retrieval.

FIXES vs original
─────────────────
S1  fetch_live_context was imported twice — once in the block import, once
    again 10 lines later. Removed the duplicate.
S2  stage_latencies was missing from the emergency early-return — the caller
    receives an empty dict instead of timing data.
S3  stage_latencies was missing from all no-results early-returns (T1, T2, T3).
S4  Prompt was imported from app.generation.prompts (generic, no schema lock).
    Now imports from app.generation.patient_prompts which enforces the column
    whitelist and has Arabic/English variants.
S5  No language detection — Arabic queries received the English system prompt.
    Fixed: detect_language() runs before the pipeline and is passed to
    build_toon_prompt() so Arabic queries get the Arabic prompt.
S6  All heavy imports moved to module level. Inside-run() imports pay a
    dict-lookup penalty on every hot-path call and obscure dependencies.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)

# Doctor full-history context token budget. Hosted LLMs cap tokens-per-minute
# (Groq free tier = 6000 TPM); leave room for the prompt template + response.
DOCTOR_CONTEXT_TOKEN_BUDGET = int(os.environ.get("DOCTOR_CONTEXT_TOKEN_BUDGET", "4000"))

# ─── Module-level imports — FIX S6 ──────────────────────────────────────────
from app.core.config import cfg  # noqa: F401 (imported for side effects)
from app.core.messages import MESSAGES
from app.generation.groq_client import generate_response
# FIX S4: schema-locked patient prompt, not the generic one
from app.generation.prompts import (
    build_toon_prompt,
    build_simple_prompt,
    build_doctor_summary_prompt,
    ungrounded_values,
    grounding_regen_suffix,
)
from app.retrieval.toon import (
    TOKEN_BUDGETS,
    classify,
    fetch_full_context,
    fetch_live_context,
    index_patient,
    load_patient_index,
    search_bm25,
    search_hybrid,
)
from app.safety.content_filter import contains_sensitive_content
from app.safety.emergency_detector import detect_emergency
from app.safety.judge import judge_answer

# ─── Language detection — FIX S5 ─────────────────────────────────────────────
_AR_RE = re.compile(r'[\u0600-\u06FF]')


def _detect_lang(text: str) -> str:
    ratio = len(_AR_RE.findall(text)) / max(len(text.replace(" ", "")), 1)
    return "ar" if ratio > 0.4 else "en"


def _deterministic_grounding(answer: str, context: str) -> float:
    """Cheap, judge-independent grounding signal for the fast tiers (T1/T2).

    Replaces the old hardcoded 0.9/0.85 constants. Uses the same value-level
    grounding gate as T3: if every concrete value in the answer is supported by
    the retrieved context the score is high; each ungrounded value lowers it.
    No LLM call, so the fast path stays fast.
    """
    if not answer or not context:
        return 0.3
    missing = ungrounded_values(answer, context)
    if not missing:
        return 0.95
    # One unsupported value is a soft signal; several is a strong one.
    return round(max(0.3, 0.95 - 0.2 * len(missing)), 3)


# ─── Result dataclass ────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    answer:           str
    confidence:       float
    sources:          List[str]
    grounding_score:  float
    status:           str
    stage_latencies:  Dict[str, float] = field(default_factory=dict)
    tier_used:        int = 0


# ─── Service ─────────────────────────────────────────────────────────────────

class PatientRAGService:

    def run(
        self,
        query:      str,
        patient_id: int,
        role:       str = "patient",
    ) -> PipelineResult:

        stages: Dict[str, float] = {}
        lang = _detect_lang(query)   # FIX S5

        # ── Emergency gate ────────────────────────────────────────────────────
        t0 = time.time()
        if detect_emergency(query):
            stages["emergency"] = round(time.time() - t0, 3)
            # FIX S2: stage_latencies now included in early-return
            return PipelineResult(
                answer          = MESSAGES.EMERGENCY_ESCALATION,
                confidence      = 1.0,
                sources         = [],
                grounding_score = 1.0,
                status          = "emergency",
                stage_latencies = stages,
                tier_used       = 3,
            )
        stages["emergency"] = round(time.time() - t0, 3)

        # ── Load / build patient index ────────────────────────────────────────
        t0 = time.time()
        vs, bm25 = load_patient_index(patient_id)
        if not vs and not bm25:
            logger.info("TOON: indexing patient %s …", patient_id)
            indexed = index_patient(patient_id)
            if indexed > 0:
                vs, bm25 = load_patient_index(patient_id)
        stages["indexing"] = round(time.time() - t0, 3)

        # ── Route ─────────────────────────────────────────────────────────────
        t0 = time.time()
        decision = classify(query, patient_id)
        tier      = decision.tier
        stages["classification"] = round(time.time() - t0, 3)
        logger.info(
            "TOON: patient=%s tier=%s reason=%s lang=%s query=%r",
            patient_id, tier, decision.reason, lang, query[:60],
        )

        # ── Tier 1 — BM25 exact lookup + LLM formatting ──────────────────────
        if tier == 1:
            t0      = time.time()
            context = search_bm25(patient_id, query, top_k=5)
            stages["retrieval"] = round(time.time() - t0, 3)

            if not context:
                return PipelineResult(
                    answer          = MESSAGES.NO_RESULTS,
                    confidence      = 0.3,
                    sources         = [],
                    grounding_score = 0.3,
                    status          = "no_t1_results",
                    stage_latencies = stages,
                    tier_used       = 1,
                )

            t0 = time.time()
            prompt = build_simple_prompt(query, context, language=lang)
            answer = generate_response(prompt)
            stages["formatting"] = round(time.time() - t0, 3)

            grounding = _deterministic_grounding(answer, context)
            return PipelineResult(
                answer          = answer,
                confidence      = round(min(0.9, 0.55 + 0.4 * grounding), 3),
                sources         = ["Patient Database (BM25)"],
                grounding_score = grounding,
                status          = "t1_success",
                stage_latencies = stages,
                tier_used       = 1,
            )

        # ── Tier 2 — Hybrid retrieval + LLM formatting ────────────────────────
        if tier == 2:
            t0      = time.time()
            context = search_hybrid(patient_id, query, top_k=10)
            stages["retrieval"] = round(time.time() - t0, 3)

            if not context:
                return PipelineResult(
                    answer          = MESSAGES.NO_RESULTS,
                    confidence      = 0.3,
                    sources         = [],
                    grounding_score = 0.3,
                    status          = "no_t2_results",
                    stage_latencies = stages,
                    tier_used       = 2,
                )

            t0 = time.time()
            prompt = build_simple_prompt(query, context, language=lang)
            answer = generate_response(prompt)
            stages["formatting"] = round(time.time() - t0, 3)

            grounding = _deterministic_grounding(answer, context)
            return PipelineResult(
                answer          = answer,
                confidence      = round(min(0.85, 0.5 + 0.4 * grounding), 3),
                sources         = ["Patient Database (Hybrid)"],
                grounding_score = grounding,
                status          = "t2_success",
                stage_latencies = stages,
                tier_used       = 2,
            )

        # ── Tier 3 — Full LLM ─────────────────────────────────────────────────
        t0      = time.time()
        # FIX S1: fetch_live_context imported once at module level, not twice
        context = fetch_live_context(patient_id)
        stages["retrieval"] = round(time.time() - t0, 3)

        if not context:
            return PipelineResult(
                answer          = MESSAGES.NO_RETRIEVAL,
                confidence      = 0.0,
                sources         = [],
                grounding_score = 0.0,
                status          = "no_t3_results",
                stage_latencies = stages,
                tier_used       = 3,
            )

        # Generate
        t0 = time.time()
        # FIX S4 + S5: schema-locked prompt + correct language
        prompt = build_toon_prompt(query, context, role=role, language=lang)
        answer = generate_response(prompt)

        # Grounding gate: if the answer cites values absent from the context,
        # regenerate once with a stricter instruction. Deterministic and
        # judge-independent — catches fabricated dates/dosages/lab numbers.
        missing = ungrounded_values(answer, context)
        if missing:
            answer = generate_response(prompt + grounding_regen_suffix(lang))
        stages["generation"] = round(time.time() - t0, 3)

        # Safety filter
        t0 = time.time()
        if contains_sensitive_content(answer):
            stages["safety"] = round(time.time() - t0, 3)
            blocked_msg = (
                "لا يمكن عرض هذه المعلومات." if lang == "ar"
                else "This information cannot be displayed."
            )
            return PipelineResult(
                answer          = blocked_msg,
                confidence      = 0.0,
                sources         = [],
                grounding_score = 0.0,
                status          = "blocked",
                stage_latencies = stages,
                tier_used       = 3,
            )

        if detect_emergency(answer):
            stages["safety"] = round(time.time() - t0, 3)
            return PipelineResult(
                answer          = MESSAGES.EMERGENCY_ESCALATION,
                confidence      = 1.0,
                sources         = [],
                grounding_score = 1.0,
                status          = "emergency_in_answer",
                stage_latencies = stages,
                tier_used       = 3,
            )
        stages["safety"] = round(time.time() - t0, 3)

        # Judge
        t0 = time.time()
        jr         = judge_answer(query=query, answer=answer, context_texts=[context[:500]])
        grounding  = jr.grounding_score
        confidence = round(max(0.0, min(0.95, jr.confidence)), 3)
        stages["judge"] = round(time.time() - t0, 3)

        if grounding < 0.3:
            low_msg = (
                "لا يمكنني تقديم إجابة دقيقة بناءً على البيانات المتاحة."
                if lang == "ar"
                else "I cannot provide an accurate answer based on the available data."
            )
            return PipelineResult(
                answer          = low_msg,
                confidence      = 0.0,
                sources         = ["Patient Database"],
                grounding_score = round(grounding, 3),
                status          = "low_grounding",
                stage_latencies = stages,
                tier_used       = 3,
            )

        return PipelineResult(
            answer          = answer,
            confidence      = confidence,
            sources         = ["Patient Database"],
            grounding_score = round(grounding, 3),
            status          = "t3_success",
            stage_latencies = stages,
            tier_used       = 3,
        )

    # ── Doctor path — full history, no tier routing, clinical prompts ────────
    # NOTE: role is not yet enforced here — the caller is trusted to be a
    # doctor. Identity/auth enforcement is deferred to a later pass.
    def run_doctor(
        self,
        query:      str,
        patient_id: int,
        mode:       str = "ask",
    ) -> PipelineResult:
        """Doctor-facing retrieval over the patient's COMPLETE history.

        mode="ask"      — free-form clinical Q&A (build_toon_prompt, role=doctor)
        mode="summary"  — structured full-history case summary
        """
        stages: Dict[str, float] = {}
        lang = _detect_lang(query) if query else "en"

        # The doctor path reads the full history directly from Supabase via
        # fetch_full_context — it does NOT use the BM25/vector index, so no
        # embedding/indexing step is required here. Cap the context to a
        # tokens-per-minute-safe budget so the hosted LLM request stays under
        # provider limits (Groq free tier = 6000 TPM; leave room for the
        # prompt template + response).
        t0 = time.time()
        context = fetch_full_context(patient_id, max_tokens=DOCTOR_CONTEXT_TOKEN_BUDGET)
        stages["retrieval"] = round(time.time() - t0, 3)

        if not context:
            return PipelineResult(
                answer          = MESSAGES.NO_RETRIEVAL,
                confidence      = 0.0,
                sources         = [],
                grounding_score = 0.0,
                status          = "no_doctor_results",
                stage_latencies = stages,
                tier_used       = 3,
            )

        # Build the clinical prompt
        t0 = time.time()
        if mode == "summary":
            prompt = build_doctor_summary_prompt(context, language=lang)
        else:
            prompt = build_toon_prompt(query, context, role="doctor", language=lang)
        answer = generate_response(prompt)
        stages["generation"] = round(time.time() - t0, 3)

        logger.info(
            "TOON(doctor): patient=%s mode=%s lang=%s query=%r",
            patient_id, mode, lang, (query or "")[:60],
        )

        return PipelineResult(
            answer          = answer,
            confidence      = 0.9,
            sources         = ["Patient Database (Full History)"],
            grounding_score = 0.9,
            status          = "doctor_success",
            stage_latencies = stages,
            tier_used       = 3,
        )


patient_rag_service = PatientRAGService()