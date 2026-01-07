# ai_client.py

import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import NotFound

print("🔥 USING ai_client.py FROM:", __file__)

# =========================
# Gemini Configuration
# =========================

_DEFAULT_GEMINI_MODEL = "models/gemini-2.0-flash"


def configure_gemini():
    """
    Loads GEMINI_API_KEY from .env and configures Gemini.
    Safe to call multiple times.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to your .env file like:\n"
            "GEMINI_API_KEY=your-key-here"
        )

    genai.configure(api_key=api_key)


# =========================
# Helpers
# =========================

def _to_gemini_contents(messages):
    """
    Convert OpenAI-style messages to Gemini format.
    """
    contents = []

    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")

        if role == "assistant":
            role = "model"
        elif role == "system":
            role = "user"
            text = f"[SYSTEM INSTRUCTION]\n{text}"

        contents.append({
            "role": role,
            "parts": [text]
        })

    return contents


# =========================
# Main Chat Function
# =========================

@retry(
    wait=wait_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(Exception),
)
def call_openai_chat(
    messages,
    model: str = _DEFAULT_GEMINI_MODEL,
    max_tokens: int = 200,
    temperature: float = 0.5,
) -> str:
    """
    Calls Gemini chat model.
    """

    # 🔐 Ensure Gemini is configured BEFORE use
    configure_gemini()

    contents = _to_gemini_contents(messages)

    def _generate(model_name: str):
        gm = genai.GenerativeModel(model_name)
        return gm.generate_content(
            contents,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

    try:
        response = _generate(model)
    except NotFound:
        fallback = "models/gemini-2.0-flash"
        response = _generate(fallback)

    if not response or not getattr(response, "text", None):
        if response and getattr(response, "candidates", None):
            parts = []
            for c in response.candidates:
                if c.content and c.content.parts:
                    for p in c.content.parts:
                        if getattr(p, "text", None):
                            parts.append(p.text)
            return "\n".join(parts).strip()
        return ""

    return response.text.strip()


# =========================
# Token Counter
# =========================

def count_tokens_for_messages(messages, model: str = _DEFAULT_GEMINI_MODEL) -> int:
    """
    Count tokens using Gemini API.
    """
    try:
        configure_gemini()
        contents = _to_gemini_contents(messages)
        gm = genai.GenerativeModel(model)
        info = gm.count_tokens(contents)
        return int(info.total_tokens)
    except Exception:
        approx = sum(len(m.get("content", "")) for m in messages) // 3
        return max(approx, 0)
