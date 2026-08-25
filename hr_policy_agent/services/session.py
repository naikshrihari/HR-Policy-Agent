"""User session tool — port of the ORA_USER_SESSION_TOOL (GET_USER_SESSION node).

In Oracle Fusion this tool returns the ``PersonNumber`` of the logged-in Team Member.
Here it is an interface with a mock default; supply the real person number through the
run configuration (see :func:`hr_policy_agent.agent.HRPolicyAgent.answer`).
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class UserSessionTool:
    def __init__(self, default_person_number: Optional[str] = None):
        self._default = default_person_number

    def get_session(self, person_number: Optional[str] = None) -> Dict[str, Any]:
        """Return the Oracle-shaped ``{"items": [{"PersonNumber": ...}]}`` payload."""
        pn = person_number or self._default
        return {"items": [{"PersonNumber": pn}]}
