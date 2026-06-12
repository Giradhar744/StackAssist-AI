"""
models/llm.py
-------------
LLM loader with automatic fallback chain (all free):
  Groq llama-3.3-70b  →  Groq gemma2-9b-it
  →  Gemini 2.0-flash  →  Gemini 2.5-flash-lite
"""

import os
from config.config import (
    get_google_api_key,
    get_groq_api_key,
    GROQ_MODELS,
    GEMINI_MODELS,
)


def _try_groq(model: str):
    key = get_groq_api_key()
    if not key:
        return None
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(model=model, api_key=key, temperature=0.2)
        llm.invoke("hi")
        print(f"✅ Using Groq: {model}")
        return llm
    except Exception as e:
        print(f"⚠️ Groq {model} failed: {e}")
        return None


def _try_gemini(model: str):
    try:
        os.environ["GOOGLE_API_KEY"] = get_google_api_key()
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=model, temperature=0.2)
        llm.invoke("hi")
        print(f"✅ Using Gemini: {model}")
        return llm
    except Exception as e:
        print(f"⚠️ Gemini {model} failed: {e}")
        return None


def get_all_llms() -> list:
    """
    Returns a list of all working LLMs in priority order.
    Used by prompt.py to try the next model when one gets rate limited mid-query.
    """
    llms = []

    for model in GROQ_MODELS:
        key = get_groq_api_key()
        if not key:
            break
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(model=model, api_key=key, temperature=0.2)
            llms.append(("groq", model, llm))
        except Exception:
            pass

    for model in GEMINI_MODELS:
        try:
            os.environ["GOOGLE_API_KEY"] = get_google_api_key()
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model=model, temperature=0.2)
            llms.append(("gemini", model, llm))
        except Exception:
            pass

    return llms


def get_llm_with_fallback():
    """
    Returns the first working LLM for initial app startup.
    """
    for model in GROQ_MODELS:
        llm = _try_groq(model)
        if llm:
            return llm

    for model in GEMINI_MODELS:
        llm = _try_gemini(model)
        if llm:
            return llm

    raise RuntimeError(
        "❌ All LLM providers failed or are rate-limited. "
        "Please check your API keys and try again later."
    )


# Keep old name as alias so nothing else breaks
def get_chatgemini_model():
    return get_llm_with_fallback()
