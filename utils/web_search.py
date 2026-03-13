# utils/web_search.py

import time
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper


class RateLimitError(Exception):
    """Raised when any API hits a rate limit."""
    pass


class WebSearchError(Exception):
    """Raised when web search fails."""
    pass


def web_search(query: str, num_results: int = 5, retries: int = 2) -> str:
    """
    Perform web search using DuckDuckGo.
    - Retries on transient errors
    - Raises RateLimitError if rate limited
    - Raises WebSearchError on other failures
    """

    for attempt in range(retries):
        try:
            search = DuckDuckGoSearchAPIWrapper()
            results = search.results(query, num_results)

            if not results:
                return "No relevant web results found."

            formatted_results = []
            for r in results:
                title = r.get("title", "No Title")
                snippet = r.get("snippet", "No snippet available")
                link = r.get("link", "")
                formatted_results.append(
                    f"Title: {title}\nSnippet: {snippet}\nSource: {link}"
                )

            return "\n\n".join(formatted_results)

        except Exception as e:
            err = str(e).lower()

            # Rate limit detection 
            if any(keyword in err for keyword in ["ratelimit", "rate limit", "429", "too many requests"]):
                raise RateLimitError(
                    f"DuckDuckGo rate limit hit. Please wait a moment and try again.\nDetails: {str(e)}"
                )

            # Retry on transient errors 
            if attempt < retries - 1:
                print(f"⚠️ Web search attempt {attempt + 1} failed: {e}. Retrying...")
                time.sleep(2)
                continue

            # Final failure 
            raise WebSearchError(f"Web search failed after {retries} attempts: {str(e)}")