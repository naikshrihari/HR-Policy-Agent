"""Deterministic offline stand-ins for the LLM nodes (the ``fake`` provider).

These are NOT a reimplementation of the prompts — they are lightweight rules that let
the whole LangGraph run, be demoed, and be unit-tested without a real model or API key.
Set ``HRPA_LLM_PROVIDER`` to a real provider (openai/anthropic/ollama) to run the actual
prompts in ``hr_policy_agent/prompts/``.
"""

from __future__ import annotations

import re
from typing import Any, Dict

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|hola|good\s+(morning|afternoon|evening)|buenos\s+d[ií]as|"
    r"buenas\s+(tardes|noches)|thanks|thank\s+you|gracias|bye|adi[oó]s)[\s!.,]*$",
    re.IGNORECASE,
)
_SPANISH_HINTS = re.compile(
    r"[¿¡áéíóúñ]|\b(hola|cómo|como|qué|que|cuánto|cuanto|política|politica|vacaciones|"
    r"pago|permiso|gracias|puedo|tengo|necesito|días|dias)\b",
    re.IGNORECASE,
)


def _detect_language(text: str) -> str:
    return "ES" if _SPANISH_HINTS.search(text or "") else "EN"


def fake_llm(code: str, state: Dict[str, Any], rendered_prompt: str = "") -> Any:
    """Return a deterministic output for LLM node ``code``."""
    nodes = state.get("nodes", {})
    user_msg = state.get("input_message", "")

    if code == "INPUT_USER_QUERY":
        return {"searchQuery": user_msg, "preservedConstraints": ""}

    if code == "INTENT_ROUTE_LLM":
        language = _detect_language(user_msg)
        intent = "GREETING" if _GREETING_RE.match(user_msg or "") else "POLICY"
        # CODE-node outputs are stored wrapped in the Oracle {"result": ...} envelope.
        person = (nodes.get("RETRIEVE_PERSON_DETAILS_SCRIPT") or {}).get("result") or {}
        tm_type = person.get("tmType", "Non Represented")
        return {"intent": intent, "tmType": tm_type, "language": language}

    if code in ("CREATE_UNIQUE_QUERY", "CREATE_UNIQUE_QUERY_SPANISH"):
        iuq = nodes.get("INPUT_USER_QUERY") or {}
        return iuq.get("searchQuery", user_msg)

    if code in ("REDIRECT_LLM", "REDIRECT_LLM_SPANISH"):
        if code.endswith("SPANISH"):
            return ("Soy el asistente de políticas de RR. HH. Puedo ayudarle con preguntas "
                    "sobre las políticas de la empresa. ¿En qué tema de RR. HH. puedo ayudarle?")
        return ("I'm the HR Policy assistant. I can help with questions about company policy. "
                "What HR topic can I help you with?")

    if code in ("ANSWER_AGENT_", "ANSWER_AGENT_SPANISH"):
        # Compose from whichever RAG branch produced a value.
        for rag_code in ("TAVERN_RAG", "REPRESENTED_RAG", "NON_REPRESENTED_RAG",
                         "REPRESENTED_RAG_SPANISH", "NON_REPRESENTED_RAG_SPANISH"):
            rag = nodes.get(rag_code)
            if isinstance(rag, dict) and rag.get("value"):
                return rag["value"]
        # No RAG (redirect path) -> pass through the redirect message.
        redirect = nodes.get("REDIRECT_LLM") or nodes.get("REDIRECT_LLM_SPANISH")
        return redirect or ""

    if code in ("FINAL_ANWER_GENERATOR", "FINAL_ANWER_GENERATOR_SPANISH"):
        answer_code = "ANSWER_AGENT_SPANISH" if code.endswith("SPANISH") else "ANSWER_AGENT_"
        return nodes.get(answer_code) or ""

    if code in ("GET_THE_RELEVANT_CITATION_ONLY", "GET_THE_RELEVANT_CITATION_ONLY_SPANISH"):
        return {"Document_Title": [], "Citation_Details": []}

    if code in ("GET_NEGATIVE_FEEDBACK_DETAILS", "GET_NEGATIVE_FEEDBACK_DETAILS_SPANISH"):
        return "Please tell us what was wrong with the answer."

    if code in ("NEGATIVE_FEEDBACK_INPUT_CHECK", "NEGATIVE_FEEDBACK_INPUT_CHECK_SPANISH"):
        # True when the user's feedback detail is a usable 1/2/3 selection or non-trivial text.
        detail = str(state.get("feedback_detail", "") or "").strip()
        return bool(detail) and detail.lower() not in ("", "n/a")

    if code in ("NEGATIVE_FEEDBACK_LLM", "NEGATIVE_FEEDBACK_LLM_SPANISH"):
        if code.endswith("SPANISH"):
            return "Gracias por sus comentarios. Los usaremos para mejorar nuestras respuestas."
        return "Thank you for your feedback. We'll use it to improve our answers."

    return ""
