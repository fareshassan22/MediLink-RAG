import os
import logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("GROQ_API_KEY")

if not API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file")

client = Groq(api_key=API_KEY)


def generate_response(prompt: str, max_tokens: int = 1024):
    try:
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
        logger.error(f"🔥 Groq Error: {str(e)}")
        return "حدث خطأ أثناء توليد الإجابة."
