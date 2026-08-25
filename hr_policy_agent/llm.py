"""LLM factory and helpers.

The original workflow ran every LLM node on Oracle's ``GPT_OSS_120B`` model.  Here the
chat model is pluggable through LangChain.  Set ``HRPA_LLM_PROVIDER`` to ``openai``,
``anthropic`` or ``ollama`` to talk to a real model; the default ``fake`` provider is a
deterministic offline stub (see :mod:`hr_policy_agent.heuristics`) that lets the whole
graph run without network access or API keys.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any, Optional

from .config import Settings


def get_chat_model(settings: Settings, temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None):
    """Instantiate a LangChain chat model for the configured provider."""
    provider = settings.llm_provider.lower()
    temp = settings.llm_temperature if temperature is None else temperature
    max_tok = settings.llm_max_tokens if max_tokens is None else max_tokens

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temp,
            max_tokens=max_tok,
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.llm_model,
            temperature=temp,
            max_tokens=max_tok,
            api_key=settings.llm_api_key,
        )
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=settings.llm_model, temperature=temp)

    raise ValueError(
        f"Unknown LLM provider {settings.llm_provider!r}. Use openai/anthropic/ollama, "
        "or keep the default 'fake' provider (handled without get_chat_model)."
    )


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(text: str) -> Any:
    """Best-effort extraction of a JSON object from an LLM response."""
    if text is None:
        return {}
    text = text.strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_RE.search(text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return {}
    return {}
