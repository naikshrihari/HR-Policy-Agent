"""Pluggable embedding models for the Chroma vector-DB backend.

Providers (set ``HRPA_EMBEDDING_PROVIDER``):

* ``openai``      — ``text-embedding-3-small`` by default. Torch-free, high quality,
                    needs an API key. **Recommended for Windows.**
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


def get_embeddings(settings: Settings):
    """Return an embeddings object for the configured provider."""
    provider = settings.embedding_provider.lower()
    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )
    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model)
    if provider == "fake":
        return DeterministicHashEmbeddings()
    raise ValueError(f"Unknown embedding provider {settings.embedding_provider!r}")
