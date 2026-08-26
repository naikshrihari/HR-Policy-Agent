"""RAG document tools — port of the 5 RAG_DOCUMENT_TOOL nodes.

Each Oracle "Document Tool" retrieves grounded passages from a handbook corpus and
returns an answer with citations following this schema (AnswerWithCitations)::

    { "value": "<answer text>", "citations": [ { "citedText": "..." } ] }

Downstream nodes additionally read ``supportingChunks`` and
``documentIdentificationCriteria.documentTitle`` when present, so the tools emit those
too.

Two implementations are provided:

* :class:`MockDocumentTool` — returns canned handbook passages so the graph runs with
  no corpus or API access (default).
* :class:`RetrieverDocumentTool` — retrieves from a LangChain retriever and (optionally)
  synthesizes the ``value`` with the configured chat model.  Wire your own vector store
  in :func:`build_document_tools`.

The five handbook tool codes map to the RAG node codes used in the graph.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional

from ..config import Settings

# RAG node code -> (handbook tool code, human label, language)
RAG_TOOLS: Dict[str, Dict[str, str]] = {
    "TAVERN_RAG": {"tool": "TAVERN_ENGLISH_DOCUMENT_TOOL_WORKFLOW_V3",
                   "title": "Tavern Team Member Handbook", "language": "EN"},
    "REPRESENTED_RAG": {"tool": "REP_ENGLISH_DOCUMENT_TOOL_WORKFLOW_V3",
                        "title": "Represented Team Member Handbook", "language": "EN"},
    "NON_REPRESENTED_RAG": {"tool": "NON_REP_ENGLISH_DOCUMENT_TOOL_WORKFLOW_V3",
                            "title": "Non-Represented Team Member Handbook", "language": "EN"},
    "REPRESENTED_RAG_SPANISH": {"tool": "REP_SPANISH_DOCUMENT_TOOL_WORKFLOW",
                                "title": "Manual del Miembro del Equipo (Representado)", "language": "ES"},
    "NON_REPRESENTED_RAG_SPANISH": {"tool": "NON_REP_SPANISH_DOCUMENT_TOOL_WORKFLOW_V3",
                                    "title": "Manual del Miembro del Equipo (No Representado)", "language": "ES"},
}


class BaseDocumentTool:
    def __init__(self, node_code: str, title: str, language: str):
        self.node_code = node_code
        self.title = title
        self.language = language

    def query(self, question: str) -> Dict[str, Any]:  # pragma: no cover - interface
        raise NotImplementedError


class MockDocumentTool(BaseDocumentTool):
    """Deterministic offline tool returning a canned handbook passage + citation."""

    def query(self, question: str) -> Dict[str, Any]:
        if self.language == "ES":
            passage = (
                "Según el Manual del Miembro del Equipo, los Miembros del Equipo elegibles "
                "acumulan Tiempo Personal Libre (PL) de acuerdo con su antigüedad y horas "
                "trabajadas. Consulte la sección de licencias del manual para conocer los "
                "requisitos de elegibilidad específicos."
            )
        else:
            passage = (
                "According to the Team Member Handbook, eligible Team Members accrue Personal "
                "Leave (PL) based on length of service and hours worked. Refer to the leave "
                "section of the handbook for the specific eligibility requirements that apply "
                "to your role."
            )
        return {
            "value": passage,
            "citations": [
                {
                    "citedText": passage,
                    "documentIdentificationCriteria": {"documentTitle": self.title},
                }
            ],
            "supportingChunks": [
                {"textChunk": passage,
                 "documentIdentificationCriteria": {"documentTitle": self.title}}
            ],
        }


class RetrieverDocumentTool(BaseDocumentTool):
    """Retrieve passages from a LangChain retriever; optionally synthesize the answer."""

    def __init__(self, node_code: str, title: str, language: str,
                 retriever: Any, answer_fn: Optional[Callable[[str, List[str]], str]] = None,
                 top_k: int = 6):
        super().__init__(node_code, title, language)
        self.retriever = retriever
        self.answer_fn = answer_fn
        self.top_k = top_k

    def query(self, question: str) -> Dict[str, Any]:
        docs = self.retriever.invoke(question)[: self.top_k]
        citations = []
        chunks = []
        texts = []
        for d in docs:
            text = getattr(d, "page_content", str(d))
            title = (getattr(d, "metadata", {}) or {}).get("documentTitle", self.title)
            texts.append(text)
            citations.append({"citedText": text,
                              "documentIdentificationCriteria": {"documentTitle": title}})
            chunks.append({"textChunk": text,
                           "documentIdentificationCriteria": {"documentTitle": title}})
        value = self.answer_fn(question, texts) if self.answer_fn else "\n\n".join(texts)
        return {"value": value, "citations": citations, "supportingChunks": chunks}


def build_document_tools(settings: Settings) -> Dict[str, BaseDocumentTool]:
    """Return {rag_node_code: DocumentTool}.

    Uses mocks unless ``HRPA_USE_MOCK_RAG=false`` and a corpus directory with a
    per-tool sub-folder exists; wire your own retriever construction here.
    """
    tools: Dict[str, BaseDocumentTool] = {}
    for node_code, meta in RAG_TOOLS.items():
        if settings.use_mock_rag:
            tools[node_code] = MockDocumentTool(node_code, meta["title"], meta["language"])
            continue
        if settings.rag_backend.lower() == "chroma":
            retriever = _chroma_retriever(meta["tool"], settings)
        else:
            corpus = os.path.join(settings.rag_corpus_dir, meta["tool"])
            retriever = _build_retriever(corpus, settings)
        if retriever is None:
            tools[node_code] = MockDocumentTool(node_code, meta["title"], meta["language"])
        else:
            tools[node_code] = RetrieverDocumentTool(
                node_code, meta["title"], meta["language"], retriever, top_k=settings.rag_top_k
            )
    return tools


def _chroma_retriever(tool_code: str, settings: Settings):  # pragma: no cover - optional
    """Return a retriever over the persisted Chroma collection for ``tool_code``.

    Returns None (→ mock fallback) if Chroma/embeddings aren't installed or the
    collection is empty (i.e. you haven't run the ingest script yet).
    """
    try:
        from langchain_chroma import Chroma
    except ImportError:
        return None
    from ..embeddings import get_embeddings

    store = Chroma(
        collection_name=tool_code,
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(settings),
    )
    try:
        if store._collection.count() == 0:
            return None
    except Exception:
        return None
    return store.as_retriever(search_kwargs={"k": settings.rag_top_k})


class _Chunk:
    """Minimal Document-compatible object (has .page_content and .metadata)."""

    __slots__ = ("page_content", "metadata")

    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


def _split_text(text: str, chunk_size: int = 1400, overlap: int = 150) -> List[str]:
    """Paragraph-aware character chunker (no external splitter dependency)."""
    text = text.replace("\r\n", "\n")
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            # Long single paragraph: hard-split with overlap.
            while len(p) > chunk_size:
                chunks.append(p[:chunk_size])
                p = p[chunk_size - overlap:]
            buf = p
    if buf:
        chunks.append(buf)
    return chunks or ([text] if text.strip() else [])


def load_chunks(corpus_dir: str, chunk_size: int = 1400, overlap: int = 150) -> List[_Chunk]:
    """Load and chunk every supported (.txt/.md/.pdf/.docx) file under ``corpus_dir``."""
    from .loaders import iter_corpus_files, load_file_text

    chunks: List[_Chunk] = []
    for path, fn in iter_corpus_files(corpus_dir):
        text = load_file_text(path)
        for piece in _split_text(text, chunk_size, overlap):
            chunks.append(_Chunk(piece, {"documentTitle": fn}))
    return chunks


# Backwards-compatible alias.
_load_chunks = load_chunks


class TfidfRetriever:
    """Lightweight TF-IDF cosine retriever (scikit-learn only — no torch/faiss).

    Exposes ``.invoke(query)`` returning the top matching chunks, so it plugs into
    :class:`RetrieverDocumentTool` exactly like a LangChain retriever.
    """

    def __init__(self, chunks: List[_Chunk], top_k: int = 6):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = chunks
        self.top_k = top_k
        self._vectorizer = TfidfVectorizer(stop_words=None)
        self._matrix = self._vectorizer.fit_transform(c.page_content for c in chunks)

    def invoke(self, query: str) -> List[_Chunk]:
        import numpy as np

        q = self._vectorizer.transform([query])
        scores = (self._matrix @ q.T).toarray().ravel()
        order = np.argsort(scores)[::-1]
        return [self.chunks[i] for i in order[: self.top_k] if scores[i] > 0]


def _build_retriever(corpus_dir: str, settings: Settings):  # pragma: no cover - optional live path
    """Build an in-memory retriever over .txt/.md/.pdf/.docx files in ``corpus_dir``.

    Prefers a torch-free scikit-learn TF-IDF retriever; falls back to a FAISS +
    sentence-transformers retriever only if scikit-learn is unavailable but those are
    installed.  Returns None if the directory is missing/empty or no backend is
    available, in which case the caller uses the mock tool.
    """
    if not os.path.isdir(corpus_dir):
        return None
    top_k = settings.rag_top_k
    chunks = load_chunks(corpus_dir, settings.rag_chunk_size, settings.rag_chunk_overlap)
    if not chunks:
        return None

    # Preferred: lightweight TF-IDF (scikit-learn only).
    try:
        import sklearn  # noqa: F401
        return TfidfRetriever(chunks, top_k=top_k)
    except ImportError:
        pass

    # Optional heavier path: FAISS + HuggingFace embeddings.
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.documents import Document
    except ImportError:
        return None
    docs = [Document(page_content=c.page_content, metadata=c.metadata) for c in chunks]
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(docs, embeddings).as_retriever()
