"""One place to choose the model, so the whole team uses the same one.

The free Gemini tier has about 20 requests a day PER MODEL. When the main model is
out of quota (429) or overloaded (503), the request moves to the next model in
GEMINI_FALLBACK_MODELS, so a run doesn't stop because one model is busy.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACKS = "gemini-3.6-flash,gemini-3.5-flash"
MAX_RETRIES = 1        # per model: fail fast and move on (the library's default waits minutes)
TIMEOUT_SECONDS = 120  # per request


def get_llm(temperature: float = 0.0):
    """Gemini chat model from the .env file (GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_FALLBACK_MODELS)."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    def make(model: str):
        return ChatGoogleGenerativeAI(model=model, temperature=temperature,
                                      max_retries=MAX_RETRIES, timeout=TIMEOUT_SECONDS)

    main = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
    backups = [m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_FALLBACKS).split(",")
               if m.strip() and m.strip() != main]
    llm = make(main)
    # with_structured_output() on the result also applies to the fallbacks
    return llm.with_fallbacks([make(m) for m in backups]) if backups else llm
