# ai_client.py

import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import NotFound

# =========================
# CONFIG
# =========================
_DEFAULT_GEMINI_MODEL = "models/gemini-2.0-flash"


# =========================
# GEMINI SETUP
# =========================
def configure_gemini():
    """
    Load API key and configure Gemini.
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
# HELPERS
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
            text = f"[SYSTEM]\n{text}"

        contents.append({"role": role, "parts": [text]})
    return contents


# =========================
# MAIN CHAT FUNCTION
# =========================
def call_openai_chat(
    messages,
    model: str = _DEFAULT_GEMINI_MODEL,
    max_tokens: int = 400,
    temperature: float = 0.3,
) -> str:
    """
    Call Gemini safely.
    Never crashes the app.
    """

    try:
        configure_gemini()

        gm = genai.GenerativeModel(model)
        response = gm.generate_content(
            _to_gemini_contents(messages),
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            },
        )

        if not response or not getattr(response, "text", None):
            return "⚠️ I couldn’t generate a response this time."

        return response.text.strip()

    except NotFound:
        return "⚠️ Model not found. Please try again later."

    except Exception as e:
        err = str(e).lower()

        if "resourceexhausted" in err or "quota" in err:
            return (
                "⚠️ NestMind is temporarily out of thinking capacity.\n\n"
                "This happens due to API usage limits.\n"
                "Please wait a minute and try again."
            )

        return f"⚠️ Unexpected error: {e}"


# =========================
# TOKEN ESTIMATION (OPTIONAL)
# =========================
def count_tokens_for_messages(messages) -> int:
    """
    Rough token estimation.
    """
    approx = sum(len(m.get("content", "")) for m in messages) // 3
    return max(approx, 0)
