"""Post-answer feedback loop (the HUMAN + feedback nodes of the workflow).

The original graph, after showing the answer, collects a thumbs-up / thumbs-down via a
HUMAN node and either logs positive feedback or asks the Team Member to pick a reason
(1/2/3), validates it, thanks them, and logs negative feedback.  That interaction spans
multiple chat turns, so it is kept out of the single-turn Q&A graph and offered here as
an explicit helper that mirrors the same routing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .config import Settings
from .heuristics import fake_llm
from .llm_nodes import run_llm_node
from .services.chat_store import ChatStoreClient

POSITIVE_RESPONSE = {"EN": "Thank you for your Feedback", "ES": "Gracias por sus comentarios"}
NEGATIVE_RESPONSE = {"EN": "Thank you for your feedback", "ES": "Gracias por sus comentarios"}
VAGUE_RESPONSE = {
    "EN": ("Invalid option selected. Please choose one of the options below.\n\n"
           "1. The answer was not accurate\n2. The answer did not fully answer my question\n"
           "3. The answer was difficult to understand\nPlease type 1, 2, or 3 to continue."),
    "ES": ("Opción inválida. Por favor elija una de las siguientes opciones.\n\n"
           "1. La respuesta no fue precisa\n2. La respuesta no respondió completamente mi pregunta\n"
           "3. La respuesta fue difícil de entender\nEscriba 1, 2 o 3 para continuar."),
}
NEGATIVE_PROMPT = {
    "EN": ("To help us understand the issue, please enter the number that best describes your "
           "concern:\n\n1. The answer was not accurate\n2. The answer did not fully answer my "
           "question\n3. The answer was difficult to understand\nPlease type 1, 2, or 3 to continue."),
    "ES": ("Para ayudarnos a entender el problema, ingrese el número que mejor describa su "
           "inquietud:\n\n1. La respuesta no fue precisa\n2. La respuesta no respondió completamente "
           "mi pregunta\n3. La respuesta fue difícil de entender\nEscriba 1, 2 o 3 para continuar."),
}


def handle_feedback(rating: str, detail: Optional[str], language: str,
                    chat_store: ChatStoreClient, settings: Settings,
                    base_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Route a feedback turn.

    ``rating`` is "APPROVED" or "REJECTED".  For a rejection, ``detail`` is the Team
    Member's 1/2/3 selection (validated the same way the workflow's LLM check does).
    Returns ``{"response": ..., "needs_detail": bool}``.  ``needs_detail`` is True when
    we still need a valid 1/2/3 selection (i.e. show the negative-feedback prompt again).
    """
    lang = "ES" if str(language).upper() == "ES" else "EN"
    payload = dict(base_payload or {})

    if str(rating).upper() == "APPROVED":
        payload.update({"feedback": "APPROVED", "feedbackComment": None})
        chat_store.positive_feedback(payload)
        return {"response": POSITIVE_RESPONSE[lang], "needs_detail": False}

    # Rejection path.
    if detail is None:
        return {"response": NEGATIVE_PROMPT[lang], "needs_detail": True}

    state = {"input_message": "", "nodes": {}, "feedback_detail": detail}
    check_code = "NEGATIVE_FEEDBACK_INPUT_CHECK_SPANISH" if lang == "ES" else "NEGATIVE_FEEDBACK_INPUT_CHECK"
    if settings.llm_provider.lower() == "fake":
        valid = bool(fake_llm(check_code, state))
    else:
        valid = bool(run_llm_node(check_code, state, settings))

    if not valid:
        return {"response": VAGUE_RESPONSE[lang], "needs_detail": True}

    payload.update({"feedback": "REJECTED", "feedbackComment": detail})
    chat_store.negative_feedback(payload)
    return {"response": NEGATIVE_RESPONSE[lang], "needs_detail": False}
