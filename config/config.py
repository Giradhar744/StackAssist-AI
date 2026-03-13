"""
config/config.py
Central configuration file for the project.
Handles API keys and application settings.
"""

import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()


# API KEY MANAGEMENT 

def get_google_api_key() -> str:
    """
    Get Google API key.

    Priority:
    1. Streamlit Secrets (production)
    2. Environment variable / .env (local development)
    """

    try:
        return st.secrets["GOOGLE_API_KEY"]
    except Exception:
        key = os.getenv("GOOGLE_API_KEY")

        if not key:
            raise EnvironmentError(
                "GOOGLE_API_KEY not found. "
                "Add it to .env file or Streamlit Secrets."
            )

        return key


#  PROJECT PATHS (IMPORTANT FIX)

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Knowledge base folder
KNOWLEDGE_BASE_PATH = os.path.join(BASE_DIR, "knowledge_base")

# Vector database storage (FAISS)
VECTOR_DB_PATH = os.path.join(BASE_DIR, "vector_db")


# Chat model
GEMINI_LLM_MODEL = "gemini-2.5-flash"


# Embedding model
HUGGINGFACE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ETRIEVAL SETTINGS 

# Number of chunks retrieved from FAISS
RETRIEVER_TOP_K = 4

BATCH_SIZE = 40
SLEEP_TIME = 6

# TEXT SPLITTING SETTINGS 

# Larger chunks reduce total embeddings → faster indexing
CHUNK_SIZE = 1200

# Overlap helps preserve context between chunks
CHUNK_OVERLAP = 200
