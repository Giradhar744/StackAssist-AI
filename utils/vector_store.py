import os
import time
import numpy as np
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from models.embeddings import get_embeddings
from config.config import VECTOR_DB_PATH, BATCH_SIZE, SLEEP_TIME


def normalize_vectors(vectors):
    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, a_min=1e-8, a_max=None)


class DummyEmbeddings(Embeddings):
    """
    Placeholder Embeddings object required by FAISS.from_embeddings().
    We pre-compute real embeddings ourselves, so this is never actually
    called during indexing — it only satisfies LangChain's type check.
    It IS used during similarity_search, so it must return the correct
    vector dimension (1536 for text-embedding-004 / OpenAI ada-002).
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [np.zeros(384, dtype=np.float32).tolist() for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        return np.zeros(384, dtype=np.float32).tolist()


def create_vector_store(chunks):
    if vector_store_exists():
        print("✅ Vector store already exists. Loading instead...")
        return load_vector_store()

    if not chunks:
        raise ValueError("No chunks provided")

    embeddings_model = get_embeddings()
    save_path = os.path.abspath(VECTOR_DB_PATH)
    os.makedirs(save_path, exist_ok=True)

    vector_store = None

    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        print(f"Processing batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

        # Compute embeddings and normalize
        batch_embeddings = [embeddings_model.embed_query(doc.page_content) for doc in batch]
        batch_embeddings = normalize_vectors(batch_embeddings)

        # FAISS expects List[Tuple[str, List[float]]]
        text_embedding_pairs = [
            (doc.page_content, emb.tolist())
            for emb, doc in zip(batch_embeddings, batch)
        ]
        metadatas = [doc.metadata for doc in batch]

        if vector_store is None:
            vector_store = FAISS.from_embeddings(
                text_embeddings=text_embedding_pairs,
                embedding=DummyEmbeddings(),   # ✅ proper Embeddings object, no more warning
                metadatas=metadatas
            )
        else:
            try:
                vector_store.add_embeddings(
                    text_embeddings=text_embedding_pairs,
                    metadatas=metadatas
                )
            except Exception as e:
                if "429" in str(e):
                    print("⚠️ Rate limit hit, waiting 20s...")
                    time.sleep(20)
                    vector_store.add_embeddings(
                        text_embeddings=text_embedding_pairs,
                        metadatas=metadatas
                    )
                else:
                    raise e

        time.sleep(SLEEP_TIME)

    vector_store.save_local(save_path)
    print(f"✅ Vector store saved at {save_path}")
    return vector_store


def load_vector_store(path=VECTOR_DB_PATH):
    if not vector_store_exists(path):
        raise FileNotFoundError(f"Vector store not found at {path}")

    embeddings_model = get_embeddings()
    vector_store = FAISS.load_local(
        path,
        embeddings=embeddings_model,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded successfully")
    return vector_store


def vector_store_exists(path=VECTOR_DB_PATH):
    index_file = os.path.join(path, "index.faiss")
    metadata_file = os.path.join(path, "index.pkl")
    return os.path.exists(index_file) and os.path.exists(metadata_file)