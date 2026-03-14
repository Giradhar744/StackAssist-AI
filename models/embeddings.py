import os
from langchain_huggingface import HuggingFaceEmbeddings
from config.config import get_google_api_key


def get_embeddings():
    try:
        # ✅ Ensure key is in environment
        os.environ["GOOGLE_API_KEY"] = get_google_api_key()

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return embeddings

    except Exception as e:
        raise RuntimeError(f"HuggingFace embedding initialization failed: {str(e)}")