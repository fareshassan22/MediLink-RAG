"""LLM-based answer judge for grounding, hallucination, and quality assessment.

Replaces the cosine-similarity grounding check and regex hallucination checker
with a single Groq API call that can reason about answer quality.
"""

import json
import logging
import os
import time
import threading
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Judge provider/model — configurable so the judge can be a DIFFERENT model
# (and different provider) than the GENERATOR. Using an independent model
# removes same-family judge bias (our grounding scores were an upper bound when
# generator==judge) AND uses a separate provider rate limit, so the judge never
# drains the generator's Groq daily-token budget.
#
#   JUDGE_PROVIDER=gemini   (default) -> Google Gemini, needs GEMINI_API_KEY
#   JUDGE_PROVIDER=groq               -> Groq, needs GROQ_API_KEY
#
# Override the model with JUDGE_MODEL (e.g. gemini-2.0-flash, gemini-1.5-flash,
# or for groq: llama-3.3-70b-versatile / gemma2-9b-it).
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "gemini").strip().lower()

_DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
}


def _resolve_model() -> str:
    """Pick the judge model, guarding against a stale JUDGE_MODEL env var that
    doesn't match the active provider (e.g. a leftover 'llama-...' value while
    JUDGE_PROVIDER=gemini, which would 404 on Gemini)."""
    default = _DEFAULT_MODELS.get(JUDGE_PROVIDER, "gemini-2.0-flash")
    requested = os.getenv("JUDGE_MODEL", "").strip()
    if not requested:
        return default
    if JUDGE_PROVIDER == "gemini" and not requested.lower().startswith(("gemini", "models/")):
        logger.warning(
            "JUDGE_MODEL=%r is not a Gemini model but JUDGE_PROVIDER=gemini; "
            "using %s instead.", requested, default,
        )
        return default
    if JUDGE_PROVIDER == "groq" and requested.lower().startswith("gemini"):
        logger.warning(
            "JUDGE_MODEL=%r is a Gemini model but JUDGE_PROVIDER=groq; "
            "using %s instead.", requested, default,
        )
        return default
    return requested


JUDGE_MODEL = _resolve_model()

_judge_client = None
_judge_lock = threading.Lock()


def _get_client():
    """Lazy-load the judge client for the configured provider (thread-safe).

    Returns the provider-specific client object, or None if its API key is
    missing.
    """
    global _judge_client
    if _judge_client is not None:
        return _judge_client

    with _judge_lock:
        if _judge_client is not None:
            return _judge_client
        from dotenv import load_dotenv

        load_dotenv()

        if JUDGE_PROVIDER == "gemini":
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not api_key:
                return None
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            _judge_client = genai.GenerativeModel(
                JUDGE_MODEL,
                system_instruction=_JUDGE_SYSTEM_PROMPT,
            )
            return _judge_client

        # Default / fallback: Groq
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        _judge_client = Groq(api_key=api_key)
        return _judge_client


def _judge_complete(client, user_prompt: str) -> str:
    """Run one judge completion and return the raw JSON text, provider-agnostic."""
    if JUDGE_PROVIDER == "gemini":
        resp = client.generate_content(
            user_prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 400,
                "response_mime_type": "application/json",
            },
        )
        return resp.text

    completion = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content


@dataclass
class JudgeResult:
    """Structured result from the judge model."""

    grounded: bool
    grounding_score: float
    has_hallucination: bool
    hallucination_risk: float
    confidence: float
    flagged_claims: List[str]
    reasoning: str
    failed: bool = False  # True when the judge call could not be completed
                          # (e.g. rate limit / parse failure) and the scores
                          # below are conservative placeholders, NOT a real
                          # measurement. Consumers must exclude these rows.


_JUDGE_SYSTEM_PROMPT = """\
You are a medical answer quality judge. You evaluate whether an AI-generated \
medical answer is grounded in the provided context.

You MUST respond with valid JSON only — no markdown, no explanation outside JSON.

Evaluation criteria:
1. GROUNDING: Is the answer's main topic and key facts supported by the context? \
Paraphrasing, summarizing, and using synonyms is ACCEPTABLE — the answer does NOT \
need to use the exact same words as the context. Arabic medical terms may differ \
in phrasing from the source text.
2. HALLUCINATION: Are there fabricated specific numbers, dosages, statistics, \
drug names, or medical claims that CONTRADICT or have NO BASIS in the context? \
General medical knowledge that is consistent with the context is NOT hallucination.
3. RELEVANCE: Does the answer address the question asked?

Scoring guidelines:
- grounding_score 0.7-1.0: Most claims are supported by or consistent with context
- grounding_score 0.4-0.7: Some claims supported, some unsupported but plausible
- grounding_score 0.0-0.4: Answer is mostly unrelated to context or contradicts it
- Only score 0.0 if the answer is completely unrelated to the context
- hallucination_risk: 0.0 (no fabrication) to 1.0 (heavily fabricated)
- confidence: 0.0 (reject answer) to 1.0 (fully trustworthy)
- If the answer says "لا تتوفر معلومات كافية" but the context DOES contain \
relevant info, set grounded=false and grounding_score=0.3.
- ONLY flag specific claims that are clearly fabricated or contradicted by context.

Return exactly this JSON structure:
{
  "grounded": true or false,
  "grounding_score": float 0-1,
  "has_hallucination": true or false,
  "hallucination_risk": float 0-1,
  "confidence": float 0-1,
  "flagged_claims": ["only clearly fabricated/contradicted claims"],
  "reasoning": "1-2 sentence explanation"
}"""


_MAX_JUDGE_CONTEXT_WORDS = 600


def _build_judge_prompt(query: str, answer: str, context_texts: List[str]) -> str:
    """Build the user prompt for the judge, trimming context to control latency."""
    # Trim each context to avoid huge prompts that slow down the judge
    trimmed = []
    total_words = 0
    for i, txt in enumerate(context_texts):
        if not txt:
            continue
        words = txt.split()
        remaining = _MAX_JUDGE_CONTEXT_WORDS - total_words
        if remaining <= 0:
            break
        if len(words) > remaining:
            words = words[:remaining]
        trimmed.append(f"[Source {i + 1}]: {' '.join(words)}")
        total_words += len(words)

    context_block = "\n---\n".join(trimmed) if trimmed else "(no context provided)"

    return f"""CONTEXT (retrieved from medical textbook):
{context_block}

QUESTION: {query}

ANSWER TO EVALUATE:
{answer}

Evaluate the answer and return JSON:"""


def _parse_judge_response(raw: str) -> Optional[JudgeResult]:
    """Parse the judge's JSON response, handling common LLM formatting issues."""
    if not raw:
        return None

    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        # Remove first line (```json or ```) and last line (```)
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end])
            except json.JSONDecodeError:
                logger.warning("Judge returned unparseable JSON: %s", text[:200])
                return None
        else:
            return None

    return JudgeResult(
        grounded=bool(data.get("grounded", False)),
        grounding_score=float(max(0.0, min(1.0, data.get("grounding_score", 0.0)))),
        has_hallucination=bool(data.get("has_hallucination", True)),
        hallucination_risk=float(
            max(0.0, min(1.0, data.get("hallucination_risk", 1.0)))
        ),
        confidence=float(max(0.0, min(1.0, data.get("confidence", 0.0)))),
        flagged_claims=data.get("flagged_claims", []),
        reasoning=str(data.get("reasoning", "")),
    )


def judge_answer(
    query: str,
    answer: str,
    context_texts: List[str],
) -> JudgeResult:
    """Evaluate an answer using the judge LLM.

    Args:
        query: The user's original question.
        answer: The generated answer to evaluate.
        context_texts: The retrieved context passages.

    Returns:
        JudgeResult with grounding, hallucination, and confidence scores.
        Falls back to conservative defaults if the API call fails.
    """
    client = _get_client()
    if client is None:
        logger.error(
            "Judge: no API key for provider '%s' — falling back to conservative defaults",
            JUDGE_PROVIDER,
        )
        return JudgeResult(
            grounded=False,
            grounding_score=0.0,
            has_hallucination=True,
            hallucination_risk=1.0,
            confidence=0.0,
            flagged_claims=[],
            reasoning="Judge unavailable — API key not configured.",
            failed=True,
        )

    user_prompt = _build_judge_prompt(query, answer, context_texts)

    max_retries = 4
    for attempt in range(max_retries):
        try:
            raw = _judge_complete(client, user_prompt)
            result = _parse_judge_response(raw)

            if result is not None:
                logger.info(
                    "Judge[%s/%s]: grounded=%s, confidence=%.2f, halluc_risk=%.2f",
                    JUDGE_PROVIDER,
                    JUDGE_MODEL,
                    result.grounded,
                    result.confidence,
                    result.hallucination_risk,
                )
                return result

            # Parse failure despite forced JSON mode — retry once more before
            # falling back, rather than giving up immediately.
            logger.warning(
                "Judge: failed to parse response (attempt %d/%d)",
                attempt + 1, max_retries,
            )
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            break

        except Exception as e:
            logger.error("Judge API call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            msg = str(e).lower()
            is_rate = any(t in msg for t in ("rate_limit", "rate limit", "quota", "429", "resource_exhausted"))
            if is_rate and attempt < max_retries - 1:
                wait = 2 ** attempt + 1  # 2s, 3s, 5s
                logger.info("Judge: rate limited, retrying in %ds...", wait)
                time.sleep(wait)
                continue
            if attempt < max_retries - 1:
                time.sleep(1)
                continue

    # Conservative fallback — don't reject but signal low confidence
    return JudgeResult(
        grounded=True,
        grounding_score=0.5,
        has_hallucination=False,
        hallucination_risk=0.3,
        confidence=0.4,
        flagged_claims=[],
        reasoning="Judge call failed — using conservative fallback.",
        failed=True,
    )
