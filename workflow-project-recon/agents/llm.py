import os
from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "120"))   # seconds per call


def get_llm(temperature: float = 0.0):
    """Single place to choose the model for every agent: local Ollama or the OpenAI API."""
    if PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=OPENAI_MODEL, temperature=temperature, timeout=LLM_TIMEOUT, max_retries=2)
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=temperature,
        num_ctx=4096,
        num_predict=768,                              # hard cap: stops runaway JSON or whitespace generation
        client_kwargs={"timeout": LLM_TIMEOUT},       # never wait forever on Ollama
    )


def model_name() -> str:
    return OPENAI_MODEL if PROVIDER == "openai" else OLLAMA_MODEL