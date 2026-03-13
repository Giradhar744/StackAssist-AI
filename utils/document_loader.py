from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
import os

def load_documents(path="knowledge_base"):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found: {path}")

        loader = DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )

        documents = loader.load()

        if not documents:
            raise ValueError("No documents found in the knowledge base.")

        return documents

    except Exception as e:
        raise RuntimeError(f"Document loading failed: {e}")