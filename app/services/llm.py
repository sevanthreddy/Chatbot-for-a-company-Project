import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_sarvam import ChatSarvam

# Load environment variables from app/.env
load_dotenv()


def get_llm():
    """
    Create and return the LLM configured through environment variables.

    Supported providers:
        - ollama
        - openai
        - sarvam

    Example .env:

        LLM_PROVIDER=ollama
        LLM_MODEL=qwen2.5:3b
    """

    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    model = os.getenv("LLM_MODEL")

    if provider == "ollama":

        return ChatOllama(model=model or "qwen2.5:3b", temperature=0)

    elif provider == "openai":

        return ChatOpenAI(model=model, temperature=0, api_key=os.getenv("llmapikey"))

    elif provider == "sarvam":

        return ChatSarvam(model=model, temperature=0, api_key=os.getenv("llmapikey"))

    else:
        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Supported providers: ollama, openai, sarvam"
        )
