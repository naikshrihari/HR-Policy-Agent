"""Pluggable embedding models for the Chroma vector-DB backend.

Providers (set ``HRPA_EMBEDDING_PROVIDER``):

* ``ollama``      — local Ollama server (e.g. ``nomic-embed-text``). Torch-free, runs
                    fully offline, no API key. **Recommended for local/private setups.**
* ``openai``      — ``text-embedding-3-small`` by default. Torch-free, high quality,
                    needs an API key.
* ``huggingface`` — local ``all-MiniLM-L6-v2``. Runs offline but pulls in torch.
* ``fake``        — deterministic hash embedding. No deps, for tests/offline demos only.
"""

from __future__ import annotations

import hashlib
from typing import List

from .config import Settings


class DeterministicHashEmbeddings:
    """A tiny, dependency-free embedding for offline testing.

    Not semantic — it hashes tokens into a fixed-width bag-of-words vector.  Good enough
    to exercise the ingest → store → retrieve path without any model or network.
    Implements the LangChain Embeddings interface (embed_documents / embed_query).
    """

    def __init__(self, size: int = 256):
        self.size = size

    def _embed(self, text: str) -> List[float]:
        vec = [0.0] * self.size
        for tok in text.lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.size] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def _require(module: str, provider: str, extra: str):
    """Import ``module`` or raise a clear, actionable error naming the pip extra."""
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"Embedding provider '{provider}' needs the '{module}' package, which isn't "
            f"installed.\n  Install it with:  pip install '.[{extra}]'\n"
            f"  (or change HRPA_EMBEDDING_PROVIDER in your .env to a provider you have.)"
        ) from exc


def get_embeddings(settings: Settings):
    """Return an embeddings object for the configured provider."""
    provider = settings.embedding_provider.lower()
    if provider == "ollama":
        mod = _require("langchain_ollama", "ollama", "vectordb-ollama")
        # When left at the OpenAI default, pick a sensible local embedding model.
        model = settings.embedding_model
        if model == "text-embedding-3-small":
            model = "nomic-embed-text"
        return mod.OllamaEmbeddings(model=model, base_url=settings.ollama_base_url)
    if provider == "openai":
        mod = _require("langchain_openai", "openai", "vectordb")
        return mod.OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    if provider == "huggingface":
        mod = _require("langchain_huggingface", "huggingface", "rag-semantic")
        return mod.HuggingFaceEmbeddings(model_name=settings.embedding_model)
    if provider == "fake":
        return DeterministicHashEmbeddings()
    raise ValueError(
        f"Unknown embedding provider {settings.embedding_provider!r}. "
        "Set HRPA_EMBEDDING_PROVIDER to one of: ollama, openai, huggingface, fake."
    )
