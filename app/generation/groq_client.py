import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client = None


def _get_client():
    """Lazy-load Groq client. Raises at call time, not import time."""
    global _client
    if _client is not None:
        return _client
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY environment variable is not set")
    _client = Groq(api_key=api_key)
    return _client


def generate_response(prompt: str, max_tokens: int = 1024):
    try:
        client = _get_client()
        completion = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "You are a safe Arabic medical assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            stop=None,
        )

        answer = completion.choices[0].message.content

        if answer:
            answer = answer.strip()

            if not answer.endswith((".", "!", "?", "؟")):
                last_period = max(
                    answer.rfind("."),
                    answer.rfind("!"),
                    answer.rfind("?"),
                    answer.rfind("؟"),
                )
                if last_period > len(answer) * 0.5:
                    answer = answer[: last_period + 1]

        return answer if answer else "حدث خطأ أثناء توليد الإجابة."

    except Exception as e:
        logger.error("Groq generation error: %s", type(e).__name__)
        return "حدث خطأ أثناء توليد الإجابة."
