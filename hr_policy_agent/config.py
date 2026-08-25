"""Central configuration for the HR Policy Agent.

The original Oracle Fusion AI Agent Studio workflow (``HR_POLICY_WORKFLOW_AGENT_V35``)
used the ``ORA_LLM_MODEL_GPT_OSS_120B`` model served through Oracle's ``CERT_BASIC``
provider.  In this Python port the model is pluggable through LangChain, so any
chat model can be substituted.  Everything is driven from environment variables so
the graph can be run without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime settings, resolved from the environment (with sensible defaults)."""

    # ----- LLM -----------------------------------------------------------------
    # Which LangChain chat-model provider to instantiate.  "openai", "anthropic",
    # "ollama" or "fake" (deterministic offline stub used for tests/demo).
    llm_provider: str = field(default_factory=lambda: os.getenv("HRPA_LLM_PROVIDER", "fake"))
    llm_model: str = field(default_factory=lambda: os.getenv("HRPA_LLM_MODEL", "gpt-4o-mini"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("HRPA_LLM_TEMPERATURE", "0.2")))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("HRPA_LLM_MAX_TOKENS", "4000")))
    llm_base_url: Optional[str] = field(default_factory=lambda: os.getenv("HRPA_LLM_BASE_URL"))
    llm_api_key: Optional[str] = field(default_factory=lambda: os.getenv("HRPA_LLM_API_KEY"))

    # ----- HCM / Oracle Fusion REST -------------------------------------------
    hcm_base_url: Optional[str] = field(default_factory=lambda: os.getenv("HRPA_HCM_BASE_URL"))
    hcm_username: Optional[str] = field(default_factory=lambda: os.getenv("HRPA_HCM_USERNAME"))
    hcm_password: Optional[str] = field(default_factory=lambda: os.getenv("HRPA_HCM_PASSWORD"))
    # When no live HCM endpoint is configured we fall back to the bundled mock.
    use_mock_hcm: bool = field(default_factory=lambda: _get_bool("HRPA_USE_MOCK_HCM", True))

    # ----- Chat store (analytics / feedback logging) --------------------------
    chat_store_url: Optional[str] = field(default_factory=lambda: os.getenv("HRPA_CHAT_STORE_URL"))
    use_mock_chat_store: bool = field(default_factory=lambda: _get_bool("HRPA_USE_MOCK_CHAT_STORE", True))

    # ----- RAG document tools --------------------------------------------------
    # Directory that holds the handbook corpora, one sub-folder per tool code.
    rag_corpus_dir: str = field(default_factory=lambda: os.getenv("HRPA_RAG_CORPUS_DIR", "data/corpus"))
    rag_top_k: int = field(default_factory=lambda: int(os.getenv("HRPA_RAG_TOP_K", "6")))
    use_mock_rag: bool = field(default_factory=lambda: _get_bool("HRPA_USE_MOCK_RAG", True))

    # ----- Human-in-the-loop feedback -----------------------------------------
    # When False the feedback branch is auto-resolved (treated as APPROVED) so the
    # graph can run non-interactively.  When True the graph interrupts for input.
    enable_human_feedback: bool = field(default_factory=lambda: _get_bool("HRPA_ENABLE_HUMAN_FEEDBACK", False))

    # Default person number used when no user session is available (mirrors the
    # "123456789" fallback in the original ENCRYPT_PERSON_NUMBER code node).
    default_person_number: str = field(default_factory=lambda: os.getenv("HRPA_DEFAULT_PERSON_NUMBER", "123456789"))


def get_settings() -> Settings:
    return Settings()
