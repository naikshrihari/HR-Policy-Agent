"""Ingest handbook documents into a persistent Chroma vector database.

Reads every ``.txt`` / ``.md`` / ``.pdf`` / ``.docx`` file under each per-tool
sub-folder of the corpus directory, chunks it, embeds the chunks with the configured
embedding model, and stores them in a Chroma collection named after the tool.  Run this
once (and again whenever the handbooks change) before serving with the Chroma backend.

Usage:
    # 1. pick an embedding provider (openai recommended on Windows — no torch)
    set HRPA_EMBEDDING_PROVIDER=openai
    set HRPA_LLM_API_KEY=sk-...            # your OpenAI key
    # 2. build the vector DB from data/corpus/<tool>/... into data/chroma
    python -m scripts.ingest

    # then serve with:
    set HRPA_USE_MOCK_RAG=false
    set HRPA_RAG_BACKEND=chroma
    python -m hr_policy_agent.cli "How much bereavement leave do I get?"

Options:
    --corpus-dir DIR   override HRPA_RAG_CORPUS_DIR
    --chroma-dir DIR   override HRPA_CHROMA_DIR
    --reset            delete each collection before re-ingesting
"""

from __future__ import annotations

import argparse
import os
import sys

from hr_policy_agent.config import get_settings
from hr_policy_agent.embeddings import get_embeddings
from hr_policy_agent.services.rag import RAG_TOOLS, load_chunks


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Embed handbooks into a Chroma vector DB.")
    parser.add_argument("--corpus-dir", help="Corpus directory (default from HRPA_RAG_CORPUS_DIR)")
    parser.add_argument("--chroma-dir", help="Chroma persist directory (default from HRPA_CHROMA_DIR)")
    parser.add_argument("--reset", action="store_true", help="Delete each collection before ingesting")
    args = parser.parse_args(argv)

    settings = get_settings()
    corpus_dir = args.corpus_dir or settings.rag_corpus_dir
    chroma_dir = args.chroma_dir or settings.chroma_dir

    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
    except ImportError:
        print("Chroma is not installed. Run:  pip install '.[vectordb]'", file=sys.stderr)
        return 1

    print(f"Embedding provider: {settings.embedding_provider} ({settings.embedding_model})")
    print(f"Corpus:  {corpus_dir}\nChroma:  {chroma_dir}\n")
    try:
        embeddings = get_embeddings(settings)
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    total = 0
    # One collection per handbook tool; a folder is named for the Oracle tool code.
    tool_codes = {meta["tool"] for meta in RAG_TOOLS.values()}
    for tool in sorted(tool_codes):
        folder = os.path.join(corpus_dir, tool)
        if not os.path.isdir(folder):
            print(f"  · {tool}: no folder, skipped")
            continue
        chunks = load_chunks(folder, settings.rag_chunk_size, settings.rag_chunk_overlap)
        if not chunks:
            print(f"  · {tool}: no documents, skipped")
            continue

        store = Chroma(collection_name=tool, persist_directory=chroma_dir,
                       embedding_function=embeddings)
        if args.reset:
            try:
                store.delete_collection()
            except Exception:
                pass
            store = Chroma(collection_name=tool, persist_directory=chroma_dir,
                           embedding_function=embeddings)

        docs = [Document(page_content=c.page_content, metadata=c.metadata) for c in chunks]
        store.add_documents(docs)
        total += len(docs)
        print(f"  ✓ {tool}: embedded {len(docs)} chunks")

    print(f"\nDone. {total} chunks stored in {chroma_dir}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
