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
        print(f"✅ Groq ready: {model}")
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
        print(f"✅ Gemini ready: {model}")
        return llm
    except Exception as e:
        print(f"⚠️ Gemini {model} failed: {e}")
        return None


def get_all_llms() -> list:
    """
    Returns a list of ALL working (smoke-tested) LLMs in priority order.
    Used by prompt.py to switch models when one gets rate limited mid-query.
    Each entry is a tuple: (provider, model_name, llm_instance)
    """
    llms = []

    for model in GROQ_MODELS:
        llm = _try_groq(model)
        if llm:
            llms.append(("groq", model, llm))

    for model in GEMINI_MODELS:
        llm = _try_gemini(model)
        if llm:
            llms.append(("gemini", model, llm))

    if not llms:
        raise RuntimeError(
            "❌ All LLM providers failed. Please check your API keys."
        )

    print(f"✅ {len(llms)} model(s) available in fallback chain")
    return llms


def get_llm_with_fallback():
    """
    Returns just the first working LLM (used for initial app startup).
    Reuses get_all_llms() so no duplicate logic.
    """
    all_llms = get_all_llms()
    _, model_name, llm = all_llms[0]
    print(f"✅ Using primary model: {model_name}")
    return llm


# Keep old name as alias so nothing else breaks
def get_chatgemini_model():
    return get_llm_with_fallback()
