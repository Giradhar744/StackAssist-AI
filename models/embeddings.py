from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """
    Initialize HuggingFace local embedding model.
    - Model: all-MiniLM-L6-v2
    - Downloads automatically on first run (~80MB)
    - Runs fully locally after that — no API key, no quota, unlimited
    - Output dimension: 384
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}  # cosine similarity ready
        )
        return embeddings

    except Exception as e:
        raise RuntimeError(
            f"HuggingFace embedding initialization failed: {str(e)}"
        )