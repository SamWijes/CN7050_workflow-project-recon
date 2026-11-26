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


def get_structured_llm(schema, temperature: float = 0.0):
    """Structured output that works reliably on both providers. Qwen on Ollama is much better at
    tool calls than raw JSON mode, so the schema is sent as a function to call."""
    if PROVIDER == "openai":
        return get_llm(temperature).with_structured_output(schema)
    return get_llm(temperature).with_structured_output(schema, method="function_calling")


def model_name() -> str:
    return OPENAI_MODEL if PROVIDER == "openai" else OLLAMA_MODEL