import os
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import GEMINI_LLM_MODEL


def get_chatgemini_model():
    """Initialize and return the Gemini chat model"""

    try:
        gemini_model = ChatGoogleGenerativeAI(
            model= GEMINI_LLM_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2
        )

        return gemini_model

    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model: {str(e)}")