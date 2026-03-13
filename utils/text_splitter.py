from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents):
    """
    Split documents into smaller chunks for embedding and retrieval.
    Uses RecursiveCharacterTextSplitter which preserves semantic structure.
    """

    try:
        # Validate input
        if not documents:
            raise ValueError("No documents provided for splitting.")

        # Initialize splitter using config values
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        # Split documents
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise RuntimeError("Text splitter returned zero chunks.")

        print(f"📄 Documents received: {len(documents)}")
        print(f"🧩 Total chunks created: {len(chunks)}")

        return chunks

    except ValueError as ve:
        raise RuntimeError(f"Input validation error: {str(ve)}")

    except Exception as e:
        raise RuntimeError(f"Error while splitting documents: {str(e)}")