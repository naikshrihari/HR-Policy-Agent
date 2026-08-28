"""Runner for the LLM nodes.

Loads the verbatim prompt for a node from ``prompts/<CODE>.txt``, resolves the
``{{$context...}}`` placeholders against graph state, then either calls the configured
chat model or (for the offline ``fake`` provider) the deterministic heuristic.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict

from .config import Settings
from .heuristics import fake_llm
from .llm import get_chat_model, parse_json
from .templating import build_context, render

_PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

# Nodes whose output is a JSON object (structured output specification in the workflow).
STRUCTURED_NODES = {
    "INTENT_ROUTE_LLM",
    "INPUT_USER_QUERY",
    "GET_THE_RELEVANT_CITATION_ONLY",
    "GET_THE_RELEVANT_CITATION_ONLY_SPANISH",
}


@lru_cache(maxsize=None)
def load_prompt(code: str) -> str:
    path = os.path.join(_PROMPT_DIR, code + ".txt")
    with open(path, encoding="utf-8") as f:
        return f.read()


@lru_cache(maxsize=None)
def _llm_config() -> Dict[str, Any]:
    path = os.path.join(_PROMPT_DIR, "_llm_config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# In fast mode, only these nodes hit the real model (with a compact prompt); every
# other LLM node is handled by the deterministic heuristic router. This turns ~6 large
# LLM calls per question into a single small one.
_FAST_MODE_REAL_NODES = {"ANSWER_AGENT_", "ANSWER_AGENT_SPANISH", "REDIRECT_LLM", "REDIRECT_LLM_SPANISH"}


def _rag_context(state: Dict[str, Any], max_chars: int = 5000) -> str:
    """Build the answer context from the RAG node's retrieved evidence.

    Prefers the supporting chunks / citations (the raw retrieved passages, each labelled
    with its source document — mirroring what the full FINAL_ANWER_GENERATOR consumes),
    and falls back to the joined ``value`` if a tool only produced that.
    """
    nodes = state.get("nodes", {})
    for code in ("TAVERN_RAG", "REPRESENTED_RAG", "NON_REPRESENTED_RAG",
                 "REPRESENTED_RAG_SPANISH", "NON_REPRESENTED_RAG_SPANISH"):
        rag = nodes.get(code)
        if not isinstance(rag, dict):
            continue
        chunks = rag.get("supportingChunks") or rag.get("citations") or []
        parts = []
        for c in chunks:
            if not isinstance(c, dict):
                continue
            text = c.get("textChunk") or c.get("citedText") or ""
            title = (c.get("documentIdentificationCriteria") or {}).get("documentTitle", "")
            text = str(text).strip()
            if text:
                parts.append(f"[{title}]\n{text}" if title else text)
        if parts:
            return "\n\n".join(parts)[:max_chars]
        if rag.get("value"):
            return str(rag["value"])[:max_chars]
    return ""


def _compact_prompt(code: str, state: Dict[str, Any], context_chars: int = 5000) -> str:
    """A short, fast prompt for the answer/redirect nodes (used in fast mode)."""
    question = (state.get("nodes", {}).get("INPUT_USER_QUERY") or {}).get(
        "searchQuery", state.get("input_message", ""))
    spanish = code.endswith("SPANISH")
    if code.startswith("REDIRECT_LLM"):
        if spanish:
            return ("Eres un asistente de políticas de RR. HH. La pregunta del usuario no trata "
                    "sobre una política de RR. HH. Redirígelo amablemente en una frase.\n\n"
                    f"Pregunta: {question}\nRespuesta:")
        return ("You are an HR policy assistant. The user's question is not about an HR policy. "
                "Politely redirect them in one sentence.\n\n"
                f"Question: {question}\nAnswer:")
    context = _rag_context(state, max_chars=context_chars)
    if spanish:
        return (
            "Eres un asistente de políticas de RR. HH. Responde la PREGUNTA del usuario usando la "
            "información relevante del CONTEXTO. La pregunta puede usar palabras distintas a la "
            "política (por ejemplo, 'el cálculo de mi nómina es incorrecto' corresponde a la "
            "política de 'Correcciones de Pago') — relaciona por SIGNIFICADO, no por palabras "
            "exactas. Da una respuesta directa en 1 o 2 frases usando solo la política que "
            "corresponde; no menciones otras políticas ni agregues detalles de fondo. SOLO si NADA "
            "en el CONTEXTO es relevante para la pregunta, responde exactamente: 'Este tema no está "
            "cubierto en los documentos de política disponibles.'\n\n"
            f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}\nRESPUESTA:")
    return (
        "You are an HR policy assistant. Answer the user's QUESTION using the relevant fact(s) in "
        "the CONTEXT. The question often uses different words than the policy (e.g. 'my payroll "
        "calculation is incorrect' corresponds to a 'Pay Corrections' policy; 'photos' relates to a "
        "'photography/recording' policy) — match by MEANING, not exact wording. Give a direct answer "
        "in ONE or TWO sentences using only the policy that applies; do not list unrelated policies "
        "or add background. ONLY if NOTHING in the CONTEXT is relevant to the question, reply "
        "exactly: 'This topic is not covered in the available policy documents.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER (one or two sentences):")


def run_llm_node(code: str, state: Dict[str, Any], settings: Settings) -> Any:
    """Execute LLM node ``code`` and return its ``$output`` value."""
    if settings.llm_provider.lower() == "fake":
        return fake_llm(code, state)

    fast = settings.fast_mode
    if fast and code not in _FAST_MODE_REAL_NODES:
        # Routing, query reformulation, citation-picking: use deterministic heuristics.
        return fake_llm(code, state)

    if fast:
        rendered = _compact_prompt(code, state, context_chars=settings.answer_context_chars)
        temperature, max_tokens = 0.1, 256
    else:
        rendered = render(load_prompt(code), build_context(state))
        cfg = _llm_config().get(code, {})
        temperature = cfg.get("temperature")
        max_tokens = cfg.get("max_tokens")

    model = get_chat_model(settings, temperature=temperature, max_tokens=max_tokens)

    from langchain_core.messages import HumanMessage

    if settings.timing:
        import sys
        import time
        start = time.time()
        response = model.invoke([HumanMessage(content=rendered)])
        print(f"[timing] {code}: {time.time() - start:.1f}s "
              f"(prompt {len(rendered)} chars)", file=sys.stderr)
    else:
        response = model.invoke([HumanMessage(content=rendered)])

    content = response.content if hasattr(response, "content") else str(response)

    if code in STRUCTURED_NODES:
        return parse_json(content)
    return content
