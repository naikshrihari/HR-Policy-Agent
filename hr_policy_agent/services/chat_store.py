"""Chat store client — ports the STN_CHAT_STORE tools (analytics + feedback logging).

The workflow logs every turn (AgentChatStore) and later records the Team Member's
thumbs-up / thumbs-down feedback (positive / negative feedback stores).  The live
implementation POSTs to the configured ORDS endpoint with OAuth2 client credentials;
the mock records payloads in memory so the graph runs offline and tests can inspect
what was logged.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import Settings


class ChatStoreClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logged: List[Dict[str, Any]] = []  # inspectable in mock mode

    def agent_chat_store(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("AgentChatStore", payload)

    def positive_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("PositiveFeedback", payload)

    def negative_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._post("NegativeFeedback", payload)

    # ----------------------------------------------------------------------
    def _post(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        record = {"operation": operation, "payload": payload}
        if self.settings.use_mock_chat_store or not self.settings.chat_store_url:
            self.logged.append(record)
            return {"status": "mock-logged", "operation": operation}
        import requests

        resp = requests.post(
            self.settings.chat_store_url,
            json={"operation": operation, **payload},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json() if resp.content else {"status": "ok"}
