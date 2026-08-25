"""High-level convenience wrapper around the compiled LangGraph."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from .config import Settings, get_settings
from .graph import Services, build_graph
from .state import AgentState


class HRPolicyAgent:
    """Station Casinos HCM Policy Agent (Workflow Agent), ported to LangGraph."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.services = Services(self.settings)
        self.graph = build_graph(self.settings, self.services)

    def run(self, message: str, person_number: Optional[str] = None,
            conversation_id: Optional[str] = None) -> AgentState:
        """Run one turn and return the full final state."""
        state: AgentState = {
            "input_message": message,
            "person_number": person_number,
            "conversation_id": conversation_id or str(uuid.uuid4()),
            "trace_id": str(uuid.uuid4()),
            "nodes": {},
        }
        return self.graph.invoke(state)

    def answer(self, message: str, person_number: Optional[str] = None,
               conversation_id: Optional[str] = None) -> str:
        """Run one turn and return just the user-visible response text (HTML)."""
        result = self.run(message, person_number, conversation_id)
        return result.get("final_response", "")
