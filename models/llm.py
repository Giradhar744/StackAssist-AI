from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import GEMINI_LLM_MODEL, get_google_api_key


def get_chatgemini_model():
    """Initialize and return the Gemini chat model"""

    try:
        gemini_model = ChatGoogleGenerativeAI(
            model=GEMINI_LLM_MODEL,
            google_api_key=get_google_api_key(),
            temperature=0.2
        )

        return gemini_model

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")