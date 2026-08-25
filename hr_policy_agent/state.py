"""LangGraph state definition.

The Oracle data-pipeline addressed every node's result through a shared context:
``{{$context.$nodes.<CODE>.$output.<path>}}``.  We reproduce that model with a
single ``nodes`` dictionary keyed by the node *code*.  Every graph node writes its
result under its own code, and downstream nodes/prompts read from it exactly the way
the original template expressions did.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, Optional
from typing_extensions import TypedDict


def merge_nodes(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Reducer that shallow-merges node outputs contributed by each graph node."""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class AgentState(TypedDict, total=False):
    # ---- Conversation-level inputs (the $context.$system / $workflow roots) ----
    input_message: str          # $context.$system.$inputMessage
    conversation_id: str        # $context.$workflow.$conversationId
    trace_id: str               # $context.$workflow.$traceId
    person_number: Optional[str]

    # ---- Node output bag (the $context.$nodes.<CODE>.$output store) ----
    nodes: Annotated[Dict[str, Any], merge_nodes]

    # ---- Human-in-the-loop feedback carried across turns ----
    feedback: Optional[str]           # "APPROVED" / "REJECTED"
    feedback_detail: Optional[str]    # free-text or 1/2/3 selection

    # ---- Terminal result surfaced to the caller ----
    final_response: str
    language: str                     # "EN" / "ES"
    routed_to_hr: bool                # topic required human HR routing
