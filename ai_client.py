# ai_client.py
import os
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv
import google.generativeai as genai
from google.api_core.exceptions import NotFound

# Load environment variables from .env
load_dotenv()

# Configure Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not set. Add it to your .env file like:\nGEMINI_API_KEY=your-key-here"
    )

genai.configure(api_key=GEMINI_API_KEY)

# Default chat model: Gemini 2.0 Flash
# (The models/ prefix avoids NotFound with some SDK versions)
_DEFAULT_GEMINI_MODEL = "models/gemini-2.0-flash"


def _to_gemini_contents(messages):
    """
    Convert OpenAI-style messages [{role:'user'|'assistant'|'system', content:'...'}]
    to Gemini 'contents' format: a list of {role, parts:[text]}
    - Gemini supports roles: 'user' and 'model' (assistant => model, system => user prefix)
    """
    contents = []
    for m in messages:
        role = m.get("role", "user")
        text = m.get("content", "")

        if role == "assistant":
            role = "model"
        elif role == "system":
            # Gemini has no 'system'; prepend a system prefix under 'user'
            role = "user"
            text = f"[SYSTEM INSTRUCTION]\n{text}"

        contents.append({"role": role, "parts": [text]})
    return contents


# Retry decorator: backoff + 3 attempts on transient errors
@retry(wait=wait_exponential(multiplier=1, max=10),
       stop=stop_after_attempt(3),
       retry=retry_if_exception_type(Exception))
def call_openai_chat(
    messages,
    model: str = _DEFAULT_GEMINI_MODEL,
    max_tokens: int = 200,
    temperature: float = 0.5
) -> str:
    """
    Backward-compatible name so you don't have to change the rest of your app.
    Internally calls Gemini with equivalent settings.
    """
    contents = _to_gemini_contents(messages)

    def _gen(model_name: str):
        gen_model = genai.GenerativeModel(model_name)
        return gen_model.generate_content(
            contents,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }
        )

    try:
        response = _gen(model)
    except NotFound:
        # Fallback to a known-good alias if the provided model name isn't found
        fallback = "models/gemini-2.0-flash"
        if model != fallback:
            response = _gen(fallback)
        else:
            raise

    # Handle candidates / safety blocks
    if not response or not getattr(response, "text", None):
        if response and getattr(response, "candidates", None):
            parts = []
            for c in response.candidates:
                if getattr(c, "content", None) and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        parts.append(getattr(p, "text", "") or "")
            if parts:
                return "\n".join([p for p in parts if p]).strip()
        return ""  # No usable text

    return response.text.strip()


def count_tokens_for_messages(messages, model: str = _DEFAULT_GEMINI_MODEL) -> int:
    """
    Count tokens for Gemini using the official count_tokens API.
    Falls back gracefully if count fails.
    """
    try:
        contents = _to_gemini_contents(messages)
        gen_model = genai.GenerativeModel(model)
        info = gen_model.count_tokens(contents)
        return int(getattr(info, "total_tokens", 0))
    except Exception:
        approx = sum(len(m.get("content", "")) for m in messages) // 3
        return max(approx, 0)
