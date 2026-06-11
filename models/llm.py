"""
models/llm.py
-------------
LLM loader with automatic fallback chain (all free):
  Cerebras llama-3.3-70b  →  Cerebras llama-3.1-8b
  →  Groq llama-3.3-70b   →  Groq gemma2-9b-it
  →  Gemini 2.0-flash      →  Gemini 2.5-flash-lite

Cerebras is called via langchain-openai's ChatOpenAI with a custom base_url
to avoid the langchain-cerebras package (which has dependency conflicts).
"""

import os
from config.config import (
    get_google_api_key,
    get_groq_api_key,
    get_cerebras_api_key,
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


def get_llm_with_fallback():
    """
    Tries each model in order and returns the first one that works.
    Order: Cerebras → Groq → Gemini (best to weakest within each).
    """



    # 1. Groq (very fast, good quality)
    for model in GROQ_MODELS:
        llm = _try_groq(model)
        if llm:
            return llm

    # 2. Gemini (reliable but rate-limited)
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
