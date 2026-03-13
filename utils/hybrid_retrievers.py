from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from config.config import RETRIEVER_TOP_K


def create_hybrid_retriever(vector_store, documents):
    """
    Creates a hybrid retriever combining
    FAISS semantic search + BM25 keyword search.
    """

    # FAISS retriever (semantic search)
    vector_retriever = vector_store.as_retriever(
        search_kwargs={"k": RETRIEVER_TOP_K}
    )

    # BM25 retriever (keyword search)
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = RETRIEVER_TOP_K

    # Hybrid ensemble
    hybrid_retriever = EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.6, 0.4]  # semantic > keyword
    )

    return hybrid_retriever
