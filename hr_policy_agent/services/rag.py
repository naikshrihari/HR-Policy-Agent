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
        corpus = os.path.join(settings.rag_corpus_dir, meta["tool"])
        retriever = _build_retriever(corpus)
        if retriever is None:
            tools[node_code] = MockDocumentTool(node_code, meta["title"], meta["language"])
        else:
            tools[node_code] = RetrieverDocumentTool(
                node_code, meta["title"], meta["language"], retriever, top_k=settings.rag_top_k
            )
    return tools


def _build_retriever(corpus_dir: str):  # pragma: no cover - optional live path
    """Build a simple in-memory retriever over .txt/.md files in ``corpus_dir``.

    Returns None if the directory is missing or LangChain community deps are absent,
    in which case the caller falls back to the mock tool.
    """
    if not os.path.isdir(corpus_dir):
        return None
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_core.documents import Document
    except ImportError:
        return None

    docs = []
    for root, _dirs, files in os.walk(corpus_dir):
        for fn in files:
            if fn.lower().endswith((".txt", ".md")):
                path = os.path.join(root, fn)
                with open(path, encoding="utf-8", errors="ignore") as f:
                    docs.append(Document(page_content=f.read(), metadata={"documentTitle": fn}))
    if not docs:
        return None
    splitter = RecursiveCharacterTextSplitter(chunk_size=1400, chunk_overlap=150)
    splits = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    store = FAISS.from_documents(splits, embeddings)
    return store.as_retriever()
