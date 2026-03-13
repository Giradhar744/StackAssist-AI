# utils/prompt.py

from langchain_core.prompts import PromptTemplate
from utils.web_search import web_search, RateLimitError, WebSearchError

# Casual greetings that should NOT trigger RAG retrieval
SMALLTALK_TRIGGERS = [
    "hi", "hello", "hey", "good morning", "good evening", "good afternoon",
    "howdy", "what's up", "sup", "greetings", "hiya", "how are you",
    "how's it going", "who are you", "what are you", "what can you do",
    "help", "thanks", "thank you", "bye", "goodbye", "see you"
]

SMALLTALK_RESPONSE = """
You are StackAssist AI, a friendly internal developer assistant for engineers.
You help developers with questions about  PostgreSQL, FastAPI, Docker, and AWS.
Respond in a warm, professional and friendly tone to this message.
Let the user know what topics you can help with.
Message: {query}
""".strip()


def is_smalltalk(query: str) -> bool:
    q = query.lower().strip().rstrip("!?.,")
    return q in SMALLTALK_TRIGGERS or (
        len(q.split()) <= 2 and any(t in q for t in SMALLTALK_TRIGGERS)
    )


def is_context_relevant(context: str, query: str) -> bool:
    if not context:
        return False

    query_keywords = set(query.lower().split())
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
        "to", "for", "of", "and", "or", "tell", "me", "about", "what",
        "how", "why", "when", "which", "si", "does", "do", "i", "who",
        "explain", "describe", "define", "overview", "introduction", "give"
    }
    query_keywords -= stop_words

    if not query_keywords:
        return True

    context_lower = context.lower()
    matched = sum(1 for kw in query_keywords if kw in context_lower)
    return (matched / len(query_keywords)) >= 0.6


def rewrite_query(llm, query: str, chat_history: list) -> str:
    """
    Rewrites vague query into specific one using conversation history.
    Falls back to original query on any failure including rate limit.
    """
    if not chat_history or len(chat_history) < 2:
        return query

    recent = chat_history[-3:]
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:200]}"
        for m in recent
    )

    rewrite_prompt = f"""You are a search query rewriter.

Given the conversation history and the current question, rewrite the question
into a clear, specific, self-contained search query.

Rules:
- Output ONLY the rewritten query — no explanation, no punctuation at the end
- Keep it under 15 words
- Include the specific topic/technology from context if the question is vague
- If the question is already specific, return it unchanged

Conversation history:
{history_text}

Current question: {query}

Rewritten query:"""

    try:
        response = llm.invoke(rewrite_prompt)
        rewritten = response.content.strip().strip('"').strip("'")
        print(f"🔄 Query rewritten: '{query}' → '{rewritten}'")
        return rewritten if rewritten else query

    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            print(f"⚠️ Rate limit on rewrite — using original query: '{query}'")
            return query  # safe fallback, don't raise here
        print(f"⚠️ Query rewrite failed — using original: {e}")
        return query


def format_web_results_for_user(query: str, web_results: str) -> str:
    """
    Formats raw web search results into a clean readable response
    when LLM is unavailable due to rate limiting.
    """
    return (
        f"⚠️ **AI response temporarily unavailable** (API limit reached).\n\n"
        f"Here are the top web results for **\"{query}\"**:\n\n"
        f"---\n\n"
        f"{web_results}\n\n"
        f"---\n"
        f"_The AI assistant will be available again shortly. "
        f"Please try again in a few minutes for a full AI-generated answer._"
    )


def get_retriever(vector_store, top_k: int = 4):
    return vector_store.as_retriever(search_kwargs={"k": top_k})


def generate_rag_response(
    llm,
    retriever,
    query: str,
    mode: str = "Concise",
    use_web_fallback: bool = True,
    chat_history: list = None
):
    """
    Full RAG pipeline with rate limit fallback to web search:
    1. Smalltalk → respond naturally
    2. Rewrite query using history (tiny LLM call)
    3. Retrieve docs from KB
    4. Check relevance → web search if needed
    5. Generate answer with Gemini
       └── If Gemini rate limited → return web search results directly
    """

    try:
        chat_history = chat_history or []

        # Step 1: Smalltalk 
        if is_smalltalk(query):
            try:
                response = llm.invoke(SMALLTALK_RESPONSE.format(query=query))
                return response.content
            except Exception as e:
                err = str(e)
                if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                    # Even for smalltalk, fall back gracefully
                    return (
                        "👋 Hi! I'm StackAssist AI.\n\n"
                        "⚠️ The AI is temporarily rate-limited. "
                        "Please try again in a few minutes!"
                    )
                raise

        # Step 2: Rewrite query (gracefully falls back on rate limit) 
        search_query = rewrite_query(llm, query, chat_history)

        # Step 3: Retrieve docs 
        docs = retriever.invoke(search_query)

        context = ""
        source = "knowledge base"

        if docs:
            context = "\n\n".join([doc.page_content for doc in docs])

        # Step 4: Web search if context not relevant 
        web_results_cache = None  # cache in case we need it for rate limit fallback

        if not is_context_relevant(context, search_query) and use_web_fallback:
            print(f"🌐 KB not relevant — web searching: '{search_query}'")
            try:
                web_results_cache = web_search(search_query)
                if web_results_cache:
                    context = f"From web search:\n\n{web_results_cache}"
                    source = "web search"
            except RateLimitError:
                raise
            except WebSearchError as e:
                print(f"⚠️ Web search failed: {e}")
                if not context:
                    context = "No relevant information found in knowledge base or web search."
                    source = "none"

        if not context:
            context = "No relevant context available."
            source = "none"

        #  Step 5: Response mode 
        if mode == "Detailed":
            instruction = "Provide a detailed answer with clear explanation and examples. Use steps if needed."
        else:
            instruction = "Provide a short, clear and friendly answer in 3-5 lines."

        # Step 6: Call Gemini — with rate limit fallback to web search
        prompt_template = """
You are StackAssist AI, a warm and helpful internal developer assistant for engineers.
You specialize in PostgreSQL, FastAPI, Docker, and AWS — but you can answer general questions too.

Guidelines:
- Always respond in a polite, clear and professional tone
- Use the provided context to answer accurately
- If context comes from web search, mention it naturally
- If you truly cannot find the answer, politely say so and suggest alternatives
- Never be dismissive — always try to be as helpful as possible

Context (source: {source}):
{context}

Question: {question}

Instruction: {instruction}

Answer:"""

        prompt = PromptTemplate(
            input_variables=["context", "question", "instruction", "source"],
            template=prompt_template,
        )

        final_prompt = prompt.format(
            context=context,
            question=query,
            instruction=instruction,
            source=source,
        )

        try:
            response = llm.invoke(final_prompt)
            return response.content

        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
                print(f"⚠️ Gemini rate limited on generation — falling back to web search")

                #  Use cached web results if already fetched 
                if web_results_cache:
                    return format_web_results_for_user(query, web_results_cache)

                # Otherwise fetch web results now
                    try:
                        fresh_web = web_search(search_query)
                        if fresh_web and "Web search error" not in fresh_web:
                            return format_web_results_for_user(query, fresh_web)
                    except Exception as web_err:
                        print(f"⚠️ Web search also failed: {web_err}")

                # Both failed — return clean message 
                return (
                    "⚠️ **AI temporarily unavailable** (API rate limit reached) "
                    "and web search fallback also failed.\n\n"
                    "Please try again in a few minutes."
                )

            raise RuntimeError(f"LLM generation failed: {err}")

    except RateLimitError:
        raise
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            raise RateLimitError(
                f"Gemini API quota exceeded. Please wait and try again.\nDetails: {err}"
            )
        raise RuntimeError(f"RAG pipeline failed: {err}")