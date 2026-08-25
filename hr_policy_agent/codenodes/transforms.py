"""Ports of the small transform code nodes.

* COMBINE_USER_QUERY_AND_QUERY_FORMULATION_CODE (+ Spanish)
* GET_THE_BEST_ANSWER (+ Spanish)
* HR_ROUTING_CLASSIFICATION (+ Spanish)
"""

from __future__ import annotations

import re
from typing import Dict


# ---------------------------------------------------------------------------
# COMBINE_USER_QUERY_AND_QUERY_FORMULATION_CODE
# ---------------------------------------------------------------------------
def combine_user_query(query: str, preserved_constraints: str, language: str = "EN") -> str:
    query = query or ""
    constraint = preserved_constraints or ""
    if language.upper() == "ES":
        if constraint.strip():
            return (
                f"Pregunta: {query} <br>RESTRICCIÓN: {constraint} <br>IMPORTANTE: "
                "Recupere únicamente la evidencia que responda directamente a la pregunta "
                "y que se adhiera estrictamente a la restricción especificada. "
            )
        return (
            f"Pregunta: {query} <br>IMPORTANTE: Recupere únicamente la evidencia que "
            "responda directamente a la pregunta. "
        )
    if constraint.strip():
        return (
            f"Question: {query} <br>CONSTRAINT: {constraint} <br>IMPORTANT: Retrieve only "
            "evidence that directly answers the question and strictly adheres to the "
            "specified constraint. "
        )
    return (
        f"Question: {query} <br>IMPORTANT: Retrieve only evidence that directly answers "
        "the question. "
    )


# ---------------------------------------------------------------------------
# GET_THE_BEST_ANSWER
# ---------------------------------------------------------------------------
_NOT_COVERED_BIGRAMS = {
    "EN": ["not covered", "isn't covered", "isnt covered"],
    "ES": ["no cubierto", "no está cubierto"],
}


def get_the_best_answer(final_answer: str, answer_agent_response: str, language: str = "EN") -> str:
    """If the final answer signals the topic is 'not covered', fall back to the
    raw answer-agent response; otherwise use the polished final answer."""
    final_answer = "" if final_answer is None else str(final_answer)
    answer_agent_response = "" if answer_agent_response is None else str(answer_agent_response)

    tokens = re.sub(r"[^\w\s']", " ", final_answer.lower()).split()
    bigrams = {f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)}
    unigrams = set(tokens)

    signal_bigrams = _NOT_COVERED_BIGRAMS["ES" if language.upper() == "ES" else "EN"]
    is_not_covered = (
        any(b in bigrams for b in signal_bigrams)
        or ({"topic", "covered", "not"} <= unigrams)
    )
    return answer_agent_response if is_not_covered else final_answer


# ---------------------------------------------------------------------------
# HR_ROUTING_CLASSIFICATION
# ---------------------------------------------------------------------------
_HR_ROUTING_TOPICS = {"CRISIS", "HARASSMENT_REPORT", "COMPLAINT", "UNION_INQUIRY"}


def hr_routing_classification(topic_matched: str) -> Dict[str, str]:
    normalized = "" if topic_matched is None else str(topic_matched).strip().upper()
    hr_routing = "1" if normalized in _HR_ROUTING_TOPICS else "0"
    return {"topic_matched": normalized, "hr_routing": hr_routing}
