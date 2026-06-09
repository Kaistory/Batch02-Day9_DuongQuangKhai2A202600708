"""Shared LLM factory for all agents.

Uses OpenAI directly via LangChain's ChatOpenAI client. The model is selected
via OPENAI_MODEL (default gpt-4o-mini).
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at the OpenAI API."""
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "1024")),
        temperature=0.3,
    )