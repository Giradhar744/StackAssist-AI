# utils/web_search.py

import time
import random
from duckduckgo_search import DDGS


class RateLimitError(Exception):
    """Raised when any API hits a rate limit."""
    pass


class WebSearchError(Exception):
    """Raised when web search fails."""
    pass


# KB topics — if query is unrelated, skip KB and go straight to web
KB_TOPICS = [
    "postgresql", "postgres", "sql", "database", "db", "query", "table", "index",
    "fastapi", "api", "endpoint", "rest", "http", "router", "pydantic",
    "docker", "container", "image", "compose", "dockerfile", "kubernetes", "k8s",
    "aws", "amazon", "ec2", "s3", "lambda", "rds", "iam", "vpc", "cloud",
    "python", "pip", "venv", "uvicorn", "gunicorn",
]


def is_kb_relevant_query(query: str) -> bool:
    """Returns True if query is likely answerable from the KB."""
    q = query.lower()
    return any(topic in q for topic in KB_TOPICS)


def web_search(query: str, num_results: int = 5, retries: int = 3) -> str:
    """
    Perform web search using DuckDuckGo DDGS with retry + backoff.
    Rotates backends to avoid rate limits.
    """

    backends = ["lite", "html", "api"]

    for attempt in range(retries):
        backend = backends[attempt % len(backends)]
        wait = 2 ** attempt + random.uniform(0, 1)  # exponential backoff

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=num_results, backend=backend))

            if not results:
                return "No relevant web results found."

            formatted_results = []
            for r in results:
                title = r.get("title", "No Title")
                snippet = r.get("body", "No snippet available")
                link = r.get("href", "")
                formatted_results.append(
                    f"**{title}**\n{snippet}\nSource: {link}"
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            err = str(e).lower()

            if any(k in err for k in ["ratelimit", "rate limit", "429", "202", "too many"]):
                if attempt < retries - 1:
                    print(f"⚠️ DuckDuckGo rate limit (backend={backend}), waiting {wait:.1f}s before retry...")
                    time.sleep(wait)
                    continue
                raise RateLimitError(
                    f"DuckDuckGo rate limit hit after {retries} attempts. Details: {str(e)}"
                )

            if attempt < retries - 1:
                print(f"⚠️ Web search attempt {attempt+1} failed: {e}. Retrying in {wait:.1f}s...")
                time.sleep(wait)
                continue

            raise WebSearchError(f"Web search failed after {retries} attempts: {str(e)}")
