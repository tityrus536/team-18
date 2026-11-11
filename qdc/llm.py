"""One place to choose the model, so the whole team uses the same one."""
from __future__ import annotations

import os

from dotenv import load_dotenv


def get_llm(temperature: float = 0.0):
    """Gemini chat model, configured from the .env file (GOOGLE_API_KEY, GEMINI_MODEL)."""
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY is missing. Copy .env.example to .env and add your key.")
    from langchain_google_genai import ChatGoogleGenerativeAI

    model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)
