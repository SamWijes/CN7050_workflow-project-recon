import os
from langchain_ollama import ChatOllama

MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def get_llm(temperature: float = 0.0):
    """Single place to configure the local model for every agent."""
    return ChatOllama(model=MODEL, temperature=temperature, num_ctx=8192)