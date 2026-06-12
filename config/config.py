"""
config/config.py
Central configuration file for the project.
Handles API keys and application settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env (local development)
load_dotenv()


# ── API KEY MANAGEMENT ─────────────────────────────────────────────

def _get_secret(key_name: str) -> str | None:
    """Helper to get a key from Streamlit secrets or environment."""
    try:
        import streamlit as st
        val = st.secrets.get(key_name, None)
        if val:
            return val
    except Exception:
        pass
    return os.getenv(key_name)


def get_google_api_key() -> str:
    key = _get_secret("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError("GOOGLE_API_KEY not found. Add it to .env or Streamlit Secrets.")
    return key


def get_groq_api_key() -> str | None:
    return _get_secret("GROQ_API_KEY")


def get_cerebras_api_key() -> str | None:
    return _get_secret("CEREBRAS_API_KEY")


# ── PROJECT PATHS ──────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")


# ── MODEL SETTINGS ─────────────────────────────────────────────────


GROQ_MODEL_PRIMARY   = "llama-3.3-70b-versatile"
GROQ_MODEL_SECONDARY = "gemma2-9b-it"

GEMINI_MODEL_PRIMARY = "gemini-2.5-flash"
GEMINI_LLM_MODEL     = "gemini-2.5-flash-lite"



GROQ_MODELS = [
    GROQ_MODEL_PRIMARY,
    GROQ_MODEL_SECONDARY,
]

GEMINI_MODELS = [
    GEMINI_MODEL_PRIMARY,
    GEMINI_LLM_MODEL,
]

# Embedding model (unchanged)
HUGGINGFACE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ── RETRIEVAL SETTINGS ─────────────────────────────────────────────

RETRIEVER_TOP_K = 4
BATCH_SIZE = 40
SLEEP_TIME = 6


# ── TEXT SPLITTING SETTINGS ────────────────────────────────────────

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
