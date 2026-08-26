"""Inspect what the RAG retriever returns for a query — a diagnostic tool.

Shows the ranked chunks (the exact text fed to the answer model) for a question, so you
can tell whether a "vague / not covered" answer is a retrieval miss (the right chunk
isn't returned) or a truncation/model issue (it is returned but ranked low).

Usage:
    python -m scripts.search "can I take photos inside the casino?"
    python -m scripts.search --tm-type "REPRESENTED" --k 8 "voting leave"

Reads the same config/.env as the app (RAG backend, embeddings, corpus).
"""

from __future__ import annotations

import argparse
import sys

from hr_policy_agent.config import get_settings
from hr_policy_agent.services.rag import RAG_TOOLS, build_document_tools

# tmType + language -> RAG node code (mirrors the graph's routing).
_RAG_BY_TYPE = {
    ("TAVERN", "EN"): "TAVERN_RAG",
    ("REPRESENTED", "EN"): "REPRESENTED_RAG",
    ("NON REPRESENTED", "EN"): "NON_REPRESENTED_RAG",
    ("REPRESENTED", "ES"): "REPRESENTED_RAG_SPANISH",
    ("NON REPRESENTED", "ES"): "NON_REPRESENTED_RAG_SPANISH",
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Inspect RAG retrieval for a query")
    parser.add_argument("query", nargs="+", help="The question to retrieve for")
    parser.add_argument("--tm-type", default="NON REPRESENTED",
                        help="TAVERN | REPRESENTED | NON REPRESENTED (default)")
    parser.add_argument("--language", default="EN", help="EN (default) | ES")
    parser.add_argument("--k", type=int, help="Number of chunks to show (default: HRPA_RAG_TOP_K)")
    parser.add_argument("--chars", type=int, default=300, help="Chars of each chunk to print")
    args = parser.parse_args(argv)

    query = " ".join(args.query)
    settings = get_settings()
    if args.k:
        settings.rag_top_k = args.k
    if settings.use_mock_rag:
        print("HRPA_USE_MOCK_RAG is true → mock retriever returns a fixed passage. "
              "Set HRPA_USE_MOCK_RAG=false to search your documents.", file=sys.stderr)

    node_code = _RAG_BY_TYPE.get((args.tm_type.upper(), args.language.upper()))
    if not node_code:
        print(f"Unknown tm-type/language: {args.tm_type} / {args.language}", file=sys.stderr)
        return 1

    tools = build_document_tools(settings)
    tool = tools[node_code]
    print(f"Backend: {settings.rag_backend if not settings.use_mock_rag else 'mock'} | "
          f"handbook: {RAG_TOOLS[node_code]['title']} | top_k={settings.rag_top_k}")
    print(f"Query: {query!r}\n")

    result = tool.query(query)
    citations = result.get("citations", [])
    if not citations:
        print("  (no chunks retrieved — the query matched nothing in this handbook)")
        return 0
    for i, c in enumerate(citations, 1):
        title = (c.get("documentIdentificationCriteria") or {}).get("documentTitle", "")
        text = " ".join(str(c.get("citedText", "")).split())
        snippet = text[: args.chars] + ("…" if len(text) > args.chars else "")
        print(f"  [{i}] ({title}) {snippet}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
