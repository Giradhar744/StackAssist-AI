"""
app.py
------
Main Streamlit application with conversation memory and rate limit handling.
"""

import streamlit as st
import os
import sys
import time

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.llm import get_chatgemini_model
from utils.vector_store import load_vector_store, create_vector_store, vector_store_exists
from utils.document_loader import load_documents
from utils.text_splitter import split_documents
from utils.prompt import generate_rag_response, RateLimitError
from utils.hybrid_retrievers import create_hybrid_retriever

from config.config import VECTOR_DB_PATH, KNOWLEDGE_BASE_PATH, RETRIEVER_TOP_K


#  Cached resource loaders
@st.cache_resource(show_spinner="Loading AI model...")
def load_llm():
    try:
        return get_chatgemini_model()
    except Exception as e:
        st.error(f"⚠️ Could not load Gemini model: {e}")
        return None


@st.cache_resource(show_spinner="Loading knowledge base index...")
def load_knowledge_base():
    try:
        if not vector_store_exists(VECTOR_DB_PATH):
            st.error(
                "⚠️ Knowledge base index not found. "
                "Please run `python build_index.py` from your terminal first."
            )
            return None

        docs = load_documents(KNOWLEDGE_BASE_PATH)
        chunks = split_documents(docs)
        vector_store = load_vector_store(VECTOR_DB_PATH)
        retriever = create_hybrid_retriever(vector_store, chunks)
        return retriever

    except Exception as e:
        st.error(f"⚠️ Failed to load knowledge base: {e}")
        return None


# Fallback plain chat (no RAG) 

def get_chat_response(chat_model, messages: list, system_prompt: str) -> str:
    try:
        formatted = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted.append(HumanMessage(content=msg["content"]))
            else:
                formatted.append(AIMessage(content=msg["content"]))
        response = chat_model.invoke(formatted)
        return response.content
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            return "⚠️ Gemini API rate limit reached. Please wait a moment and try again."
        return f"⚠️ Error getting response: {err}"


# Pages

def chat_page():
    st.title("🛠️ StackAssist AI")
    st.caption("Your internal AI assistant for  PostgreSQL, FastAPI, Docker, and AWS.")

    with st.sidebar:
        st.header("⚙️ Settings")

        response_mode = st.radio(
            "Response Mode",
            options=["Concise", "Detailed"],
            index=0,
            help=(
                "Concise: Short, direct answers (3-5 lines). "
                "Detailed: Step-by-step explanations with examples."
            ),
        )

        use_web_search = st.toggle(
            "🌐 Web Search Fallback",
            value=True,
            help="Searches the web when the knowledge base doesn't have relevant context.",
        )

        st.divider()

        if st.button("🔄 Rebuild Knowledge Index", use_container_width=True):
            load_knowledge_base.clear()
            st.success("Cache cleared. Restart the app to rebuild.")
            st.rerun()

        st.divider()

        if vector_store_exists(VECTOR_DB_PATH):
            st.success("✅ Index loaded")
        else:
            st.warning("⚠️ Index not built yet")

    llm = load_llm()
    retriever = load_knowledge_base()

    if llm is None:
        st.error(
            "The AI model failed to load. "
            "Please check your GOOGLE_API_KEY in your environment variables."
        )
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask anything about PostgreSQL, FastAPI, Docker or AWS..."):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        total_start = time.time()

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    if retriever is not None:
                        response = generate_rag_response(
                            llm=llm,
                            retriever=retriever,
                            query=prompt,
                            mode=response_mode,
                            use_web_fallback=use_web_search,
                            chat_history=st.session_state.messages[:-1],
                        )
                    else:
                        system_prompt = (
                            "You are StackAssist AI, a helpful internal developer assistant "
                            "specialising in PostgreSQL, FastAPI, Docker, and AWS. "
                            "Answer clearly and honestly. If you're not sure, say so."
                        )
                        response = get_chat_response(
                            llm, st.session_state.messages, system_prompt
                        )

                # Rate limit — show friendly warning
                except RateLimitError as e:
                    response = (
                        "⚠️ **API rate limit reached.**\n\n"
                        "The Gemini or search API is temporarily unavailable due to quota limits. "
                        "Please wait a moment and try again.\n\n"
                        f"_Details: {str(e)}_"
                    )
                    st.warning("Rate limit hit — please wait before sending another message.")

                except RuntimeError as e:
                    response = f"⚠️ Something went wrong: {str(e)}"

            total_latency = time.time() - total_start
            st.markdown(response)
            st.caption(f"⚡ Response time: {total_latency:.2f}s")

        st.session_state.messages.append({"role": "assistant", "content": response})


def instructions_page():
    st.title("📖 Setup & Usage Guide")
    st.markdown("Welcome! Here's everything you need to get StackAssist AI running.")

    st.markdown("""
    ## 🔧 Installation

    ```bash
    pip install -r requirements.txt
    ```

    ## 🔑 Environment Variables

    Create a `.env` file in the project root:

    ```
    GOOGLE_API_KEY=your_google_api_key_here
    ```

    ## 📚 Knowledge Base

    Place your documentation files (PDF, TXT, MD) into the `knowledge_base/` folder.
    Then build the index once:

    ```bash
    python build_index.py
    ```

    ## 💬 Using the Chat

    - **Concise mode**: Short, direct answers in 3-5 lines.
    - **Detailed mode**: Step-by-step explanations with examples.
    - **Web Search Fallback**: Searches the web when local docs are insufficient.
    - **Conversation Memory**: Query rewriting uses last 3 messages for context.
    """)


# Entry point

def main():
    st.set_page_config(
        page_title="StackAssist AI",
        page_icon="🛠️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("🧭 Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)

        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if page == "Instructions":
        instructions_page()
    elif page == "Chat":
        chat_page()


if __name__ == "__main__":
    main()