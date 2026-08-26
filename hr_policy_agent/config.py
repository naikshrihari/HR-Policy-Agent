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

# Load a project-root .env file (if present) so settings persist across shells and you
# don't have to re-export environment variables in every terminal. Real environment
# variables still win over .env values.
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv(override=False)
except ImportError:  # python-dotenv not installed → just use real env vars
    pass


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
    # Ollama context window (num_ctx). Big enough to hold the prompt without truncation.
    ollama_num_ctx: int = field(default_factory=lambda: int(os.getenv("HRPA_OLLAMA_NUM_CTX", "8192")))

    # Fast mode: handle routing/query-reformulation with deterministic logic and give the
    # single answer call a compact prompt, so only ~1 (small) LLM call runs per question
    # instead of 5-6 large ones. Big speedup for local models; slightly less nuanced
    # routing. Recommended when serving with Ollama on CPU.
    fast_mode: bool = field(default_factory=lambda: _get_bool("HRPA_FAST_MODE", False))
    # Print per-LLM-node timings to stderr (diagnostics).
    timing: bool = field(default_factory=lambda: _get_bool("HRPA_TIMING", False))

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
    # Retrieval backend when mock RAG is off: "tfidf" (in-memory, scikit-learn) or
    # "chroma" (persistent vector database, requires an embedding model).
    rag_backend: str = field(default_factory=lambda: os.getenv("HRPA_RAG_BACKEND", "tfidf"))
    chroma_dir: str = field(default_factory=lambda: os.getenv("HRPA_CHROMA_DIR", "data/chroma"))
    # Chunking used by both the TF-IDF loader and the ingest script.
    rag_chunk_size: int = field(default_factory=lambda: int(os.getenv("HRPA_RAG_CHUNK_SIZE", "1400")))
    rag_chunk_overlap: int = field(default_factory=lambda: int(os.getenv("HRPA_RAG_CHUNK_OVERLAP", "150")))

    # ----- Embeddings (Chroma backend) ----------------------------------------
    embedding_provider: str = field(default_factory=lambda: os.getenv("HRPA_EMBEDDING_PROVIDER", "openai"))
    embedding_model: str = field(default_factory=lambda: os.getenv("HRPA_EMBEDDING_MODEL", "text-embedding-3-small"))
    # Local Ollama server (used by the "ollama" LLM and embedding providers).
    ollama_base_url: str = field(default_factory=lambda: os.getenv("HRPA_OLLAMA_BASE_URL", "http://localhost:11434"))

    # ----- Human-in-the-loop feedback -----------------------------------------
    # When False the feedback branch is auto-resolved (treated as APPROVED) so the
    # graph can run non-interactively.  When True the graph interrupts for input.
    enable_human_feedback: bool = field(default_factory=lambda: _get_bool("HRPA_ENABLE_HUMAN_FEEDBACK", False))

    # Default person number used when no user session is available (mirrors the
    # "123456789" fallback in the original ENCRYPT_PERSON_NUMBER code node).
    default_person_number: str = field(default_factory=lambda: os.getenv("HRPA_DEFAULT_PERSON_NUMBER", "123456789"))


def get_settings() -> Settings:
    return Settings()
